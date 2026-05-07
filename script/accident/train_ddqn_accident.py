"""
train_ddqn_accident_baseline.py

DDQN Baseline under high-demand pressure with stable TensorBoard logging.
Using SB3 DQN (which implements Double DQN by default).
"""

import os, multiprocessing, random, torch, numpy as np
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from generate_train_rou import generate_training_route_with_variation
from gongguanroundaboutenv import GongguanRoundaboutEnv

# ────────── Reproducibility ──────────
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ────────── Generate Route File ──────────
SEED_FOR_ROUTE = np.random.randint(0, 1_000_000)
ROUTE_FILE, _ = generate_training_route_with_variation(
    seed=SEED_FOR_ROUTE,
    demand_scaling=1.0,   # High-demand pressure training
    noise_level=0.05
)
print(f"[INFO] DDQN Baseline Accident Training ROUTE_FILE: {ROUTE_FILE}")

# ────────── Environment Factory ──────────
def make_env(seed=None):
    def _init():
        env = GongguanRoundaboutEnv(
            sumo_cfg_path=str(_PROJECT_ROOT / "cfg" / "gongguanacc.sumocfg"),
            route_file=ROUTE_FILE,
            gui=False,
            accident=True,
            acc_lane=None,
            acc_start=600,
            acc_duration=900,
            seed=seed,
        )
        env.reset()
        return Monitor(env)
    return _init

# ────────── TensorBoard Callback (record on episode end) ──────────
class TensorboardCallback(BaseCallback):
    KEYS = (
        "avg_queue_len", "avg_delay_step", "avg_delay_perveh",
        "avg_speed", "avg_travel_time",
        "avg_step_reward", "episode_reward", "avg_phase_duration",
    )
    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info or "episode_reward" in info:
                for k in self.KEYS:
                    if k in info:
                        self.logger.record(f"custom/{k}", info[k])
                lr = self.model.policy.optimizer.param_groups[0]["lr"]
                self.logger.record("train/learning_rate", lr)
                self.logger.dump(self.num_timesteps)
        return True

# ────────── Main Training ──────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    TOTAL_TS = 512_000

    vec_env = DummyVecEnv([make_env(GLOBAL_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
    )

    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=10_000,
        batch_size=128,
        tau=0.01,
        gamma=0.995,
        train_freq=1,               
        gradient_steps=1,          
        target_update_interval=1000,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256]),  
        tensorboard_log="./tb_roundabout_accident",
        device="cuda",
        verbose=1,
        seed=GLOBAL_SEED,
    )

    callbacks = [TensorboardCallback()]

    model.learn(
        total_timesteps=TOTAL_TS,
        callback=callbacks,
        tb_log_name="ddqn_accident_baseline_high_demand",
    )

    model.save("ddqn_accident_baseline_high_demand.zip")
    vec_env.save("vec_norm_ddqn_accident_baseline_high_demand.pkl")

    print("\n[✔] DDQN high-demand accident baseline training completed.\n")
