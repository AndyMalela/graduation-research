# GongguanRoundaboutEnv
from __future__ import annotations

from generate_train_rou import generate_training_route_with_variation

import socket, time, random
from typing import Dict, List, Set

import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
from sumolib import checkBinary

# ===== 固定參數 =====
TLS_ID = "joinedS_1936977761_2037722652_2037722663_2047590029_#2more"

APPROACH_GROUPS: Dict[str, List[str]] = {
    "N":  ["222682222#1_0", "222682222#1_1", "222682222#1_2"],
    "E":  ["34241530#0_0",  "34241530#0_1"],
    "SE": ["-55029244#0_0"],
    "S":  ["-28231434_0"],
    "W":  ["198876045#1_0", "198876045#1_1"],
    "NW": ["193253433#2_0", "193253433#2_1"],
}

RING_LANES = [
    "1276692362_0", "1276692362_1", "1276692362_2",
    "1276692365_0", "1276692365_1", "1276692365_2",
    "1276692366_0", "1276692366_1", "1276692366_2",
    "308724899_0",  "308724899_1",  "308724899_2",
]

EXIT_LANES = [
    "193253427#0_0", "193253427#0_1",
    "198876043#0_0", "198876043#0_1", "198876043#0_2",
    "28231434_0",
    "23241821#0_0", "23241821#0_1",
    "198876042#0_0", "198876042#0_1", "198876042#0_2",
]

GREEN_PHASES = [0, 2, 4]
YELLOW_PHASE = {0: 1, 2: 3, 4: 5}
YELLOW_DUR = 3.0
MIN_HOLD, MAX_HOLD = 15.0, 30.0
FIXED_CYCLE_PERIOD = 30.0
FLOW_NORM = 8.0
SWITCH_PEN_START, SWITCH_PEN_END = -0.2, -0.5
CURRICULUM_STEPS = 150_000

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

class GongguanRoundaboutEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sumo_cfg_path: str,
        *,
        route_file: str,
        accident: bool = True,
        acc_lane: str | None = None,
        acc_start: int = 600,
        acc_duration: int = 900,
        seed: int | None = None,
        end_time: int = 3600,
        gui: bool = False,
        fixed_cycle: bool = False,
        demand_scaling: float = 0.8,  

    ):
        super().__init__()
        self.sumo_cfg = sumo_cfg_path
        self.route_file = route_file
        self._demand_scaling = demand_scaling

        self.accident, self.acc_lane = accident, acc_lane
        self.acc_start, self.acc_duration = acc_start, acc_duration
        self.end_time, self.gui = end_time, gui
        self.fixed_cycle = fixed_cycle
        self._rng = np.random.default_rng(seed)
        self._global_step = 0

        self.n_green_phases = len(GREEN_PHASES)
        self.action_space = spaces.Discrete(self.n_green_phases)
        obs_dim = len(APPROACH_GROUPS) * 3 + 3 + self.n_green_phases + 2
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        self._global_step = 0
        self._step_count = 0
        self._accident_triggered = False
        self._accident_cleared = False
        self.prev_vehicle_lane = {}
        self.unique_vehicles: Set[str] = set()
        self.enter_time: Dict[str, float] = {}
    
        try:
            traci.close()
            time.sleep(0.05)
        except Exception:
            pass
    
        # ===== 每集生成不同 .rou.xml =====
        gen_seed = np.random.randint(0, 1_000_000)
        self.route_file, _ = generate_training_route_with_variation(
            seed=gen_seed,
            demand_scaling=self._demand_scaling,
            noise_level=0.05
        )
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        print(f"[Env] Generated new route_file: {self.route_file} with seed {gen_seed}")
    
        # ===== 啟動 SUMO =====
        traci.start([
            checkBinary("sumo-gui" if self.gui else "sumo"),
            "-c", self.sumo_cfg,
            "--route-files", self.route_file,
            "--quit-on-end",
        ], port=_free_port())
    
        self.current_green_idx = 0
        self.current_phase = GREEN_PHASES[self.current_green_idx]
        traci.trafficlight.setPhase(TLS_ID, self.current_phase)
        self._last_switch_time = traci.simulation.getTime()
        self._lock_until = self._last_switch_time + MIN_HOLD
        self._pending_green = self._yellow_until = None
    
        self.ep_reward = 0.0
        self.sum_queue_len = 0.0
        self.sum_delay_step = 0.0
        self.total_wait_time = 0.0
        self.sum_travel_time = 0.0
        self.veh_exited = 0
        self.speed_history: List[float] = []
        self.phase_durations: List[float] = []
        self.switch_count = 0
    
        return self._get_obs(), {}
        
    def step(self, action: int):
        now = traci.simulation.getTime()

        # Accident trigger and recovery
        if self.accident and not self._accident_triggered and now >= self.acc_start:
            acc_lane = self.acc_lane or random.choice(RING_LANES)
            traci.lane.setDisallowed(acc_lane, ["passenger", "motorcycle", "truck"])
            self._accident_triggered = True
            self._acc_lane_used = acc_lane
            print(f"[Accident] Lane {acc_lane} closed at {now:.1f}s.")
        if self.accident and self._accident_triggered and not self._accident_cleared and now >= self.acc_start + self.acc_duration:
            traci.lane.setAllowed(self._acc_lane_used, ["passenger", "motorcycle", "truck"])
            self._accident_cleared = True
            print(f"[Recovery] Lane {self._acc_lane_used} reopened at {now:.1f}s.")

        # Reroute
        if self._global_step % 30 == 0:
            for vid in traci.vehicle.getIDList():
                try:
                    traci.vehicle.rerouteTraveltime(vid)
                except traci.TraCIException:
                    continue

        # Phase switching logic
        if self.fixed_cycle:
            if now >= self._last_switch_time + FIXED_CYCLE_PERIOD:
                self.current_green_idx = (self.current_green_idx + 1) % self.n_green_phases
                tgt_green = GREEN_PHASES[self.current_green_idx]
                traci.trafficlight.setPhase(TLS_ID, tgt_green)
                self.phase_durations.append(now - self._last_switch_time)
                self.current_phase = tgt_green
                self._last_switch_time = now
        else:
            if self._pending_green is not None and now >= self._yellow_until:
                traci.trafficlight.setPhase(TLS_ID, self._pending_green)
                self.phase_durations.append(now - self._last_switch_time)
                self.current_phase = self._pending_green
                self.current_green_idx = GREEN_PHASES.index(self.current_phase)
                self._last_switch_time = now
                self._pending_green = self._yellow_until = None
            if now >= self._lock_until:
                tgt_green = GREEN_PHASES[action]
                if tgt_green != self.current_phase:
                    traci.trafficlight.setPhase(TLS_ID, YELLOW_PHASE[self.current_phase])
                    traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_DUR)
                    self._pending_green = tgt_green
                    self._yellow_until = now + YELLOW_DUR
                    self._lock_until = self._yellow_until + MIN_HOLD
                    self.switch_count += 1

        traci.simulationStep()
        self._global_step += 1
        self._step_count += 1

        print(f"🤖 AI chose Action: {action}")
        print(f"👀 AI observed State (first 5 values): {obs[:5]}")
        print(f"💰 Environment gave Reward: {reward}")
        print("-" * 30)

        return self._compute_rewards_and_obs(now)

    def _compute_rewards_and_obs(self, now):
        vids = traci.vehicle.getIDList()
        self.unique_vehicles.update(vids)
        curr_wait = sum(traci.vehicle.getWaitingTime(v) for v in vids)
        self.total_wait_time += curr_wait
        delay_step = curr_wait / max(len(vids), 1)
        self.sum_delay_step += delay_step

        queue_len_step = sum(
            traci.lane.getLastStepHaltingNumber(l)
            for ls in APPROACH_GROUPS.values()
            for l in ls
        )
        self.sum_queue_len += queue_len_step

        speeds = [traci.vehicle.getSpeed(v) for v in vids]
        self.speed_history.append(sum(speeds) / max(len(speeds), 1))

        departed = traci.simulation.getDepartedIDList()
        for vid in departed:
            self.enter_time[vid] = now

        exited = 0
        for vid in vids:
            try:
                lane_id = traci.vehicle.getLaneID(vid)
                if vid in self.prev_vehicle_lane:
                    prev_lane = self.prev_vehicle_lane[vid]
                    if prev_lane not in EXIT_LANES and lane_id in EXIT_LANES:
                        exited += 1
                        enter_t = self.enter_time.pop(vid, None)
                        if enter_t is not None:
                            self.sum_travel_time += now - enter_t
                            self.veh_exited += 1
                self.prev_vehicle_lane[vid] = lane_id
            except traci.TraCIException:
                continue

        r_speed = np.mean(speeds) / 8.0 if speeds else 0.0
        r_outflow = np.clip(exited / 10.0, 0.0, 1.0)
        slow_ratio = len([v for v in speeds if v < 2.0]) / max(len(speeds), 1) if speeds else 0.0
        r_slow = -slow_ratio
        max_q = max(sum(traci.lane.getLastStepVehicleNumber(l) for l in ls) for ls in APPROACH_GROUPS.values())
        r_qmax = -np.tanh(max_q / 20.0)
        frac = min(self._global_step / CURRICULUM_STEPS, 1.0)
        r_switch = SWITCH_PEN_START + (SWITCH_PEN_END - SWITCH_PEN_START) * frac if self.switch_count > 0 else 0.0

        reward =  0.5*r_outflow + r_speed + r_slow + r_qmax + r_switch
        self.ep_reward += reward
        done = now >= self.end_time

        info = {}
        if done:
            n = max(self._step_count, 1)
            veh_total = max(len(self.unique_vehicles), 1)
        
            # ========== 修正 avg_phase_duration 異常 ==========
            if len(self.phase_durations) == 0:
                # 無切換，使用 MAX_HOLD 或實際持續時間作為合理 fallback
                fallback_duration = min(MAX_HOLD, now - self._last_switch_time)
                self.phase_durations.append(fallback_duration)
            else:
                # 有切換但最後一次未記錄，補記最後持續時間
                last_duration = now - self._last_switch_time
                if last_duration > 0:
                    self.phase_durations.append(last_duration)
            avg_phase_duration = np.mean(self.phase_durations)
            # ==========================================
        
            info.update(
                avg_queue_len=self.sum_queue_len / n,
                avg_delay_step=self.sum_delay_step / n,
                avg_delay_perveh=self.total_wait_time / veh_total,
                avg_speed=np.mean(self.speed_history),
                avg_travel_time=self.sum_travel_time / max(self.veh_exited, 1),
                avg_step_reward=self.ep_reward / n,
                episode_reward=self.ep_reward,
                avg_phase_duration=avg_phase_duration,
            )

        return self._get_obs(), reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        obs: List[float] = []
        for lanes in APPROACH_GROUPS.values():
            occ = np.mean([traci.lane.getLastStepOccupancy(l) for l in lanes]) / 100.0
            veh_cnt = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
            flow = np.clip((veh_cnt / max(len(lanes), 1)) / FLOW_NORM, 0.0, 1.0)
            speed = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in lanes]) / 8.0
            obs.extend([occ, flow, speed])
    
        obs.append(np.mean([traci.lane.getLastStepOccupancy(l) for l in RING_LANES]) / 100.0)
        ring_speeds = [traci.lane.getLastStepMeanSpeed(l) for l in RING_LANES]
        if ring_speeds:
            p25, p75 = np.percentile(ring_speeds, [25, 75])
            obs.extend([np.clip(p25 / 8.0, 0.0, 1.0), np.clip(p75 / 8.0, 0.0, 1.0)])
        else:
            obs.extend([0.0, 0.0])
    
        # 移除 _remaining_green
    
        mask = np.zeros(self.n_green_phases, dtype=np.float32)
        mask[self.current_green_idx] = 1.0
        obs.extend(mask.tolist())
    
        obs.append(np.clip((traci.simulation.getTime() - self._last_switch_time) / MAX_HOLD, 0.0, 1.0))
        obs.append(np.clip(self._step_count / self.end_time, 0.0, 1.0))
        return np.array(obs, dtype=np.float32)

    def render(self):
        if not self.gui:
            t = traci.simulation.getTime() - self._last_switch_time
            print(f"[{traci.simulation.getTime():.1f}s] phase={self.current_phase} timer={t:.1f}s")

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
