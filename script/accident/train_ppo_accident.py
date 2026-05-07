import os, multiprocessing, random, torch, numpy as np
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from gongguanroundaboutenv import GongguanRoundaboutEnv
from generate_train_rou import generate_training_route_with_variation

# ────────── Reproducibility ──────────
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ────────── Generate Route File ──────────
SEED_FOR_ROUTE = np.random.randint(0, 1_000_000)
ROUTE_FILE, _ = generate_training_route_with_variation(
    seed=SEED_FOR_ROUTE,
    demand_scaling=1,
    noise_level=0.05
)
print(f"[INFO] PPO Baseline Accident Training ROUTE_FILE: {ROUTE_FILE}")

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

# ────────── Custom Callbacks ──────────
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

class KLAdaptiveLR(BaseCallback):
    def __init__(self, thr=0.01, factor=0.5, lr_min=1e-5, lr_max=1e-3):
        super().__init__()
        self.thr, self.factor, self.lr_min, self.lr_max = thr, factor, lr_min, lr_max
    def _on_rollout_end(self):
        kl = self.logger.name_to_value.get("train/approx_kl", 0.0)
        if kl == 0.0:
            return
        opt = self.model.policy.optimizer
        cur_lr = opt.param_groups[0]["lr"]
        new_lr = cur_lr * self.factor if kl > self.thr else cur_lr / self.factor
        new_lr = max(min(new_lr, self.lr_max), self.lr_min)
        for g in opt.param_groups:
            g["lr"] = new_lr
        self.model.learning_rate = new_lr
        print(f"[KL-LR] approx_kl={kl:.5f} → lr={new_lr:.2e}")
    def _on_step(self): return True

# ────────── Main Training ──────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    N_ENVS = 4
    N_STEPS = 256
    TOTAL_TS = 512_000

    vec_env = DummyVecEnv([make_env(GLOBAL_SEED + i) for i in range(N_ENVS)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,
        clip_reward=20.0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,         
        n_steps=N_STEPS,
        batch_size=256,
        n_epochs=4,
        gamma=0.997,                 
        gae_lambda=0.95,             
        ent_coef=0.1,                
        vf_coef=1.0,                 
        clip_range=0.2,
        max_grad_norm=0.5,
        tensorboard_log="./tb_roundabout_accident",
        device="cuda",
        verbose=1,
        seed=GLOBAL_SEED,
        policy_kwargs=dict(net_arch=[128]),
    )

    callbacks = [
        TensorboardCallback(),
        KLAdaptiveLR(),
    ]

    model.learn(
        total_timesteps=TOTAL_TS,
        callback=callbacks,
        tb_log_name="ppo_accident_baseline_tuned",
    )

    model.save("ppo_accident_baseline_tuned.zip")
    vec_env.save("vec_norm_ppo_accident_baseline_tuned.pkl")

    print("\n[✔] PPO accident baseline (tuned) training completed.\n")
