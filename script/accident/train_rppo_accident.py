"""
train_rppo_accident.py
Train Recurrent PPO on GongguanRoundaboutEnv under accident scenario with reroute support using SB3-contrib.
Stable KL-adaptive LR, entropy decay, and high-demand stress testing.
"""

import os, random, torch, numpy as np, multiprocessing
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from generate_train_rou import generate_training_route_with_variation
from gongguanroundaboutenv import GongguanRoundaboutEnv

# ────────── Reproducibility ──────────
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# ────────── Generate Route File ──────────
SEED_FOR_ROUTE = np.random.randint(0, 1_000_000)
ROUTE_FILE = generate_training_route_with_variation(
    seed=SEED_FOR_ROUTE,
    demand_scaling=1.0,
    noise_level=0.05
)[0]
print(f"[INFO] Using ROUTE_FILE={ROUTE_FILE}, SEED={SEED_FOR_ROUTE}")

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

# ────────── TensorBoard Callback ──────────
class TensorboardCallback(BaseCallback):
    KEYS = (
        "avg_queue_len", "avg_delay_step", "avg_delay_perveh",
        "avg_speed", "avg_travel_time",
        "avg_step_reward", "episode_reward", "avg_phase_duration",
    )
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_reward" not in info:
                continue
            for k in self.KEYS:
                if k in info:
                    self.logger.record(f"custom/{k}", info[k])
        lr = self.model.policy.optimizer.param_groups[0]["lr"]
        self.logger.record("train/learning_rate", lr)
        return True

# ────────── FreezeNorm Callback ──────────
class FreezeNormCallback(BaseCallback):
    def __init__(self, freeze_at=150_000):
        super().__init__()
        self.freeze_at = freeze_at
        self.frozen = False
    def _on_step(self):
        if not self.frozen and self.num_timesteps >= self.freeze_at:
            self.training_env.training = False
            self.frozen = True
            print(f"[FreezeNorm] Frozen at {self.num_timesteps} steps")
        return True

# ────────── Entropy Decay Callback ──────────
class EntropyDecayCallback(BaseCallback):
    def __init__(self, final_step=200_000, start=0.02, end=0.004):
        super().__init__()
        self.final_step, self.start, self.end = final_step, start, end
    def _on_step(self):
        frac = min(self.num_timesteps / self.final_step, 1.0)
        new_ent = self.start + (self.end - self.start) * frac
        self.model.ent_coef = new_ent
        if hasattr(self.model, "ent_coef_tensor"):
            self.model.ent_coef_tensor.fill_(new_ent)
        return True

# ────────── KL Adaptive LR Callback ──────────
class KLAdaptiveLR(BaseCallback):
    def __init__(self, thr=0.02, factor=0.7, lr_min=2e-5, lr_max=2e-4):
        super().__init__()
        self.thr, self.factor, self.lr_min, self.lr_max = thr, factor, lr_min, lr_max
    def _on_rollout_end(self):
        kl = self.logger.name_to_value.get("train/approx_kl", 0.0)
        if kl == 0.0:
            return
        opt = self.model.policy.optimizer
        cur = opt.param_groups[0]["lr"]
        if kl > self.thr:
            new_lr = cur * self.factor
        else:
            new_lr = cur / self.factor
        new_lr = max(min(new_lr, self.lr_max), self.lr_min)
        for g in opt.param_groups:
            g["lr"] = new_lr
        self.model.learning_rate = new_lr
        print(f"[KL-LR] approx_kl={kl:.5f} → lr={new_lr:.2e}")
    def _on_step(self):
        return True

# ────────── Main ──────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    N_ENVS = 4
    N_STEPS = 512
    TOTAL_TS = 512_000

    vec_env = DummyVecEnv([make_env(GLOBAL_SEED + i) for i in range(N_ENVS)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,
        clip_reward=20.0,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=N_STEPS,
        batch_size=512,
        n_epochs=6,
        gamma=0.997,
        gae_lambda=0.95,
        ent_coef=0.02,
        vf_coef=0.5,
        clip_range=0.08,
        clip_range_vf=0.1,
        target_kl=0.04,
        max_grad_norm=0.3,
        policy_kwargs=dict(
            lstm_hidden_size=512,
            net_arch=dict(pi=[256], vf=[256]),
        ),
        tensorboard_log="./tb_roundabout_accident",
        device="cuda",
        verbose=1,
        seed=GLOBAL_SEED,
    )

    callbacks = [
        TensorboardCallback(),
        FreezeNormCallback(),
        EntropyDecayCallback(),
        KLAdaptiveLR(),
    ]

    model.learn(
        total_timesteps=TOTAL_TS,
        callback=callbacks,
        tb_log_name="rppo_accident_highdemand_stable",
    )

    model.save("rppo_accident_highdemand_stable.zip")
    vec_env.save("vec_norm_rppo_accident_highdemand_stable.pkl")

    print(f"\n[✔] RPPO high-demand accident training completed.\n")
