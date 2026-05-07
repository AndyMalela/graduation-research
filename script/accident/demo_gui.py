"""
demo_gui.py

Run the trained RPPO agent (or fixed-time baseline) in sumo-gui.
Live metrics are printed to the terminal every 60 simulation steps (≈ 1 sim-minute).

Usage:
    cd script/accident
    python demo_gui.py                        # RPPO, accident scenario
    python demo_gui.py --scenario peak        # RPPO, high-demand only
    python demo_gui.py --fixed-cycle          # fixed-time baseline, accident scenario
    python demo_gui.py --seed 1234            # different traffic seed
"""

import argparse
import numpy as np
import traci
from pathlib import Path

_DIR  = Path(__file__).resolve().parent
_ROOT = _DIR.parents[1]

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gongguanroundaboutenv import (
    GongguanRoundaboutEnv, APPROACH_GROUPS, GREEN_PHASES,
)
from generate_train_rou import generate_training_route_with_variation

MODEL_PATH = _DIR / "rppo_accident_highdemand_stable.zip"
NORM_PATH  = _DIR / "vec_norm_rppo_accident_highdemand_stable.pkl"
SUMO_CFG   = _ROOT / "cfg" / "gongguanacc.sumocfg"

REPORT_EVERY = 60   # steps between metric prints (≈ 1 sim-minute)
PHASE_NAMES  = {0: "Phase-0 (N/W green)", 2: "Phase-1 (E/SE green)", 4: "Phase-2 (S/NW green)"}


def _live_metrics(env: GongguanRoundaboutEnv) -> dict:
    """Read current per-step metrics directly from SUMO via TraCI."""
    sim_time  = traci.simulation.getTime()
    vids      = traci.vehicle.getIDList()
    n_veh     = max(len(vids), 1)

    speeds    = [traci.vehicle.getSpeed(v) for v in vids]
    avg_speed = np.mean(speeds) if speeds else 0.0
    slow_pct  = 100 * len([s for s in speeds if s < 2.0]) / n_veh

    queue = sum(
        traci.lane.getLastStepHaltingNumber(l)
        for ls in APPROACH_GROUPS.values()
        for l in ls
    )
    per_approach = {
        arm: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        for arm, lanes in APPROACH_GROUPS.items()
    }

    avg_q_so_far = env.sum_queue_len / max(env._step_count, 1)
    avg_r_so_far = env.ep_reward    / max(env._step_count, 1)
    avg_spd_so_far = np.mean(env.speed_history) if env.speed_history else 0.0

    return dict(
        sim_time=sim_time,
        n_veh=len(vids),
        avg_speed=avg_speed,
        slow_pct=slow_pct,
        queue_now=queue,
        per_approach=per_approach,
        avg_q_so_far=avg_q_so_far,
        avg_r_so_far=avg_r_so_far,
        avg_spd_so_far=avg_spd_so_far,
        phase=PHASE_NAMES.get(env.current_phase, str(env.current_phase)),
        switches=env.switch_count,
        accident_on=env._accident_triggered and not env._accident_cleared,
    )


def _print_metrics(m: dict, step: int) -> None:
    acc_tag = "  *** ACCIDENT ACTIVE ***" if m["accident_on"] else ""
    per_app = "  ".join(f"{arm}:{q:>2}" for arm, q in m["per_approach"].items())

    print(
        f"\n── Step {step:>5}  |  Sim time {m['sim_time']:>6.0f}s{acc_tag}\n"
        f"   Phase      : {m['phase']}  (switches so far: {m['switches']})\n"
        f"   Vehicles   : {m['n_veh']:>4}  on-network\n"
        f"   Speed now  : {m['avg_speed']:>5.2f} m/s   ({m['slow_pct']:.1f}% slow < 2 m/s)\n"
        f"   Queue now  : {m['queue_now']:>4} halted  [{per_app}]\n"
        f"   ── Episode averages so far ──\n"
        f"   Avg speed  : {m['avg_spd_so_far']:>5.2f} m/s\n"
        f"   Avg queue  : {m['avg_q_so_far']:>5.1f} vehicles\n"
        f"   Avg reward : {m['avg_r_so_far']:>+6.3f}"
    )


def _print_summary(info: dict) -> None:
    print("\n" + "=" * 50)
    print("  EPISODE SUMMARY")
    print("=" * 50)
    labels = {
        "avg_queue_len":    "Avg queue length (veh)",
        "avg_speed":        "Avg speed        (m/s)",
        "avg_travel_time":  "Avg travel time  (s)  ",
        "avg_step_reward":  "Avg step reward       ",
        "avg_phase_duration":"Avg phase duration(s) ",
        "avg_delay_perveh": "Avg delay per veh (s) ",
    }
    for k, label in labels.items():
        if k in info:
            print(f"  {label}: {info[k]:.3f}")
    print("=" * 50)


def make_gui_env(accident: bool, fixed_cycle: bool, demand_scaling: float, seed: int):
    route_file, _ = generate_training_route_with_variation(
        seed=seed, demand_scaling=demand_scaling, noise_level=0.05,
    )
    return GongguanRoundaboutEnv(
        sumo_cfg_path=str(SUMO_CFG),
        route_file=route_file,
        gui=True,
        accident=accident,
        acc_lane=None,
        acc_start=600,
        acc_duration=900,
        seed=seed,
        demand_scaling=demand_scaling,
        fixed_cycle=fixed_cycle,
    )


def run_with_model(env: GongguanRoundaboutEnv) -> None:
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(str(NORM_PATH), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = RecurrentPPO.load(str(MODEL_PATH), env=vec_env, device="cpu")

    obs = vec_env.reset()
    lstm_states   = None
    episode_starts = np.ones((1,), dtype=bool)
    done  = False
    step  = 0
    infos = [{}]

    print("\n[Demo] RPPO agent running. Terminal prints every 60 steps (≈ 1 sim-minute).")
    print("[Demo] Use SUMO GUI controls to pause / change speed.\n")

    try:
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_starts, deterministic=True,
            )
            obs, _, dones, infos = vec_env.step(action)
            episode_starts = dones
            done = bool(dones[0])
            step += 1

            if step % REPORT_EVERY == 0:
                _print_metrics(_live_metrics(env), step)

        _print_summary(infos[0])

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted.")
    finally:
        vec_env.close()


def run_fixed_cycle(env: GongguanRoundaboutEnv) -> None:
    obs, _ = env.reset()
    done   = False
    step   = 0
    info   = {}

    print("\n[Demo] Fixed-cycle controller running. Terminal prints every 60 steps.\n")

    try:
        while not done:
            obs, _, done, _, info = env.step(0)
            step += 1
            if step % REPORT_EVERY == 0:
                _print_metrics(_live_metrics(env), step)

        _print_summary(info)

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["accident", "peak"], default="accident",
                        help="accident: lane closure at t=600s;  peak: 130%% demand, no closure")
    parser.add_argument("--fixed-cycle", action="store_true",
                        help="Use fixed-time baseline instead of trained RPPO")
    parser.add_argument("--seed", type=int, default=42, help="Traffic demand seed")
    args = parser.parse_args()

    accident      = args.scenario == "accident"
    demand_scaling = 1.0 if accident else 1.3

    print(f"\n[Demo] Scenario   : {args.scenario}")
    print(f"[Demo] Controller : {'Fixed-Time (30s/phase)' if args.fixed_cycle else 'RPPO (trained model)'}")
    print(f"[Demo] Demand     : {demand_scaling}x baseline  |  seed={args.seed}")
    if accident:
        print("[Demo] Accident   : random ring lane blocked at t=600s for 15 min\n")

    env = make_gui_env(accident=accident, fixed_cycle=args.fixed_cycle,
                       demand_scaling=demand_scaling, seed=args.seed)

    if args.fixed_cycle:
        run_fixed_cycle(env)
    else:
        run_with_model(env)


if __name__ == "__main__":
    main()
