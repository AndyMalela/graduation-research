"""
train_fixed_cycle_accident_baseline.py

Fixed-Time Control Baseline Training Script (Accident Scenario Supported, Optimized)
"""

import os, numpy as np, torch, random, multiprocessing
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from gongguanroundaboutenv import GongguanRoundaboutEnv

# ────────── Reproducibility ──────────
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ────────── Custom TensorBoard Callback ──────────
class TensorboardCallback(BaseCallback):
    KEYS = (
        "avg_queue_len", "avg_delay_step", "avg_delay_perveh",
        "avg_speed", "avg_travel_time",
        "avg_step_reward", "episode_reward", "avg_phase_duration",
    )
    def __init__(self, tb_log_dir):
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(tb_log_dir)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_reward" not in info and "episode" not in info:
                continue
            for k in self.KEYS:
                if k in info:
                    # Correct step assignment
                    self.writer.add_scalar(f"custom/{k}", info[k], self.num_timesteps)
            self.writer.flush()
        return True

    def _on_training_end(self):
        self.writer.close()

# ────────── Environment Factory ──────────
def make_env(seed_offset):
    def _init():
        env = GongguanRoundaboutEnv(
            sumo_cfg_path=str(_PROJECT_ROOT / "cfg" / "gongguanacc.sumocfg"),
            route_file="dummy.rou.xml",
            gui=False,
            accident=True,
            acc_lane=None,
            acc_start=600,
            acc_duration=900,
            seed=GLOBAL_SEED + seed_offset,
            fixed_cycle=True,
            demand_scaling=1,
            end_time=1800,
        )
        return Monitor(env)
    return _init

# ────────── Fixed-Time Controller Runner ──────────
def run_fixed_cycle_baseline(total_timesteps=512_000, n_envs=4):
    tb_log_dir = os.path.join("./tb_roundabout_accident", "fixed_cycle_accident_baseline")
    os.makedirs(tb_log_dir, exist_ok=True)
    callback = TensorboardCallback(tb_log_dir)

    env_fns = [make_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
    )

    obs = vec_env.reset()
    step = 0

    while step < total_timesteps:
        action = [0] * n_envs  # Fixed phase action for all environments
        obs, reward, done, infos = vec_env.step(action)
        step += n_envs

        # Fix: explicitly update num_timesteps before calling _on_step
        callback.num_timesteps = step
        callback.locals = {"infos": infos}
        callback._on_step()

        if any(done):
            obs = vec_env.reset()

    callback._on_training_end()
    vec_env.close()
    print(f"[✔] Fixed-Time baseline completed for {total_timesteps} timesteps using {n_envs} parallel environments.")

# ────────── Main ──────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_fixed_cycle_baseline(total_timesteps=512_000, n_envs=4)
