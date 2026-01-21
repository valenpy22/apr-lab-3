import os
import json
import time
import gc
from datetime import datetime
import pandas as pd

import numpy as np
#import optuna

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from cubesat_detumbling_rl import CubeSatDetumblingEnv


# =========================
# Env factory
# =========================
def make_env(max_steps=400, granularity=40, time_step=0.1, seed=0, log_dir=None, start_time=None):
    def _init():
        env = CubeSatDetumblingEnv(
            render_mode=None,
            max_steps=max_steps,
            granularity=granularity,
            time_step=time_step,
            start_time=start_time,
            debug=False,
            plot_hist=False
        )
        env.reset(seed=seed)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        else:
            env = Monitor(env)

        return env
    return _init


def evaluate_with_history(
    model_path: str,
    n_eval_episodes: int = 50,
    seed: int = 999,
    max_steps: int = 400,
    granularity: int = 40,
    time_step: float = 1.0,
):
    """
    Corre evaluación y retorna un dict con historiales por episodio.
    Compatible con DummyVecEnv (1 env).
    """

    model_path = os.path.abspath(model_path)

    if not model_path.endswith(".zip") and os.path.isfile(model_path + ".zip"):
        model_path += ".zip"

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"El modelo no fue encontrado en {model_path}")

    load_path = model_path[:-4] if model_path.endswith(".zip") else model_path

    eval_env = DummyVecEnv([make_env(
        max_steps=max_steps, granularity=granularity, time_step=time_step,
        seed=seed, log_dir=None
    )])

    model = DQN.load(load_path)

    rewards = []
    success = []
    lengths = []
    times = []
    final_w_norm = []
    steps_to_success = []

    t_eval_start = time.time()

    for ep in range(n_eval_episodes):
        ep_start = time.time()
        obs = eval_env.reset()
        done = [False]
        ep_reward = 0.0
        steps = 0
        last_info = None

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)

            ep_reward += float(reward[0])
            steps += 1
            last_info = infos[0]  # info del único env

        ep_time = time.time() - ep_start
        print(f"[EVAL] Episode {ep+1}/{n_eval_episodes} "
              f"steps={steps} time={ep_time:.2f}s")
        # Al terminar el episodio, last_info trae tu info + monitor
        s = bool(last_info.get("success", False))
        w = float(last_info.get("angular_velocity_norm", np.nan))

        # Monitor añade episode: {r, l, t} si env está envuelto en Monitor
        ep_info = last_info.get("episode", {})
        l = int(ep_info.get("l", steps))
        t = float(ep_info.get("t", np.nan))

        rewards.append(ep_reward)
        success.append(s)
        lengths.append(l)
        times.append(t)
        final_w_norm.append(w)

        if s:
            steps_to_success.append(steps)

    eval_env.close()

    total_time = time.time() - t_eval_start
    print(f"[EVAL] TOTAL TIME: {total_time:.2f}s "
          f"(avg {total_time/n_eval_episodes:.2f}s/episode)")

    history = {
        "rewards": np.array(rewards, dtype=float),
        "success": np.array(success, dtype=bool),
        "lengths": np.array(lengths, dtype=int),
        "times": np.array(times, dtype=float),
        "final_w_norm": np.array(final_w_norm, dtype=float),
        "steps_to_success": np.array(steps_to_success, dtype=int),
        "n_eval_episodes": n_eval_episodes,
        "success_rate": float(np.mean(success)) if len(success) else 0.0
    }
    return history

def plot_eval_history_scatter(
    history: dict,
    save_dir: str | None = None,
    prefix: str = "eval"
):
    rewards = history["rewards"]
    success = history["success"]
    wnorm = history["final_w_norm"]
    steps_to_success = history["steps_to_success"]
    n = history["n_eval_episodes"]

    episodes = np.arange(1, n + 1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 1) Reward por episodio (scatter)
    # =========================
    plt.figure()
    plt.scatter(episodes, rewards, alpha=0.7)
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Evaluación: Reward por episodio")
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_reward_scatter.png"), dpi=200, bbox_inches="tight")

    # =========================
    # 2) Success rate acumulada (línea, se deja)
    # =========================
    plt.figure()
    success_cum = np.cumsum(success.astype(int)) / episodes
    plt.plot(episodes, success_cum)
    plt.xlabel("Episodio")
    plt.ylabel("Success rate")
    plt.title(f"Evaluación: Success rate acumulada (final={success_cum[-1]*100:.2f}%)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_success_rate.png"), dpi=200, bbox_inches="tight")

    # =========================
    # 3) Norma final de velocidad angular (scatter)
    # =========================
    plt.figure()
    plt.scatter(episodes, wnorm, alpha=0.7)
    plt.xlabel("Episodio")
    plt.ylabel("||ω|| final [rad/s]")
    plt.title("Evaluación: Norma final de velocidad angular")
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_final_omega_norm_scatter.png"), dpi=200, bbox_inches="tight")

    # =========================
    # 4) Histograma pasos al éxito (se mantiene)
    # =========================
    if len(steps_to_success) > 0:
        plt.figure()
        plt.hist(steps_to_success, bins=15)
        plt.xlabel("Pasos al éxito")
        plt.ylabel("Frecuencia")
        plt.title("Evaluación: Distribución de pasos al éxito")
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}_steps_to_success_hist.png"), dpi=200, bbox_inches="tight")

    plt.close("all")


def plot_training_monitor(
    monitor_csv_path: str,
    window: int = 50,
    save_dir: str | None = None,
    prefix: str = "training",
):
    """
    Lee monitor.csv de Stable-Baselines3 y grafica:
      1) Reward por episodio + media móvil
      2) Largo del episodio
      3) Reward vs tiempo de entrenamiento

    Parameters
    ----------
    monitor_csv_path : str
        Ruta al archivo monitor.csv
    window : int
        Ventana para media móvil del reward
    save_dir : str | None
        Carpeta donde guardar las figuras (si None, solo muestra)
    prefix : str
        Prefijo de los nombres de archivos
    """

    if not os.path.isfile(monitor_csv_path):
        raise FileNotFoundError(f"No existe: {monitor_csv_path}")

    # ---- Leer CSV ignorando comentarios ----
    df = pd.read_csv(
        monitor_csv_path,
        comment="#"
    )

    # Asegura columnas correctas
    assert {"r", "l", "t"}.issubset(df.columns), df.columns

    rewards = df["r"].to_numpy()
    lengths = df["l"].to_numpy()
    times = df["t"].to_numpy()

    episodes = np.arange(1, len(rewards) + 1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 1) Reward por episodio
    # =========================
    plt.figure()
    plt.plot(episodes, rewards, alpha=0.4, label="Reward por episodio")

    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            np.arange(window, len(rewards) + 1),
            ma,
            linewidth=2,
            label=f"Media móvil ({window})"
        )

    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Entrenamiento: Reward por episodio")
    plt.grid(True)
    plt.legend()

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_reward.png"),
            dpi=200,
            bbox_inches="tight"
        )

    # =========================
    # 2) Largo del episodio
    # =========================
    plt.figure()
    plt.plot(episodes, lengths)
    plt.xlabel("Episodio")
    plt.ylabel("Steps")
    plt.title("Entrenamiento: Largo del episodio")
    plt.grid(True)

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_episode_length.png"),
            dpi=200,
            bbox_inches="tight"
        )

    # =========================
    # 3) Reward vs tiempo
    # =========================
    plt.figure()
    plt.plot(times / 60.0, rewards, alpha=0.5)
    plt.xlabel("Tiempo de entrenamiento [min]")
    plt.ylabel("Reward")
    plt.title("Entrenamiento: Reward vs tiempo")
    plt.grid(True)

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_reward_vs_time.png"),
            dpi=200,
            bbox_inches="tight"
        )

    plt.close("all")

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_final_eval_from_json(
    json_path: str,
    save_dir: str | None = None,
    prefix: str = "final_eval"
):
    """
    Genera gráficos a partir del campo `final_eval`
    del dqn_results.json
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    eval_data = data["final_eval"]

    rewards = np.array(eval_data["rewards"])
    success = np.array(eval_data["success"], dtype=int)
    lengths = np.array(eval_data["lengths"])
    wnorm = np.array(eval_data["final_w_norm"])
    steps_success = np.array(eval_data["steps_to_success"])
    n = eval_data["n_eval_episodes"]

    episodes = np.arange(1, n + 1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 1) Reward por episodio
    # =========================
    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Evaluación final: Reward por episodio")
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_reward.png"), dpi=200, bbox_inches="tight")

    # =========================
    # 2) Success rate acumulada
    # =========================
    plt.figure()
    success_cum = np.cumsum(success) / episodes
    plt.plot(episodes, success_cum)
    plt.ylim(0, 1.05)
    plt.xlabel("Episodio")
    plt.ylabel("Success rate")
    plt.title(f"Evaluación final: Success rate acumulada ({success_cum[-1]*100:.1f}%)")
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_success_rate.png"), dpi=200, bbox_inches="tight")

    # =========================
    # 3) Norma final de velocidad angular
    # =========================
    plt.figure()
    plt.plot(episodes, wnorm)
    plt.xlabel("Episodio")
    plt.ylabel("||ω|| final [rad/s]")
    plt.title("Evaluación final: Norma de velocidad angular")
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_final_omega_norm.png"), dpi=200, bbox_inches="tight")

    # =========================
    # 4) Histograma pasos al éxito
    # =========================
    if len(steps_success) > 0:
        plt.figure()
        plt.hist(steps_success, bins=15)
        plt.xlabel("Pasos al éxito")
        plt.ylabel("Frecuencia")
        plt.title("Evaluación final: Distribución de pasos al éxito")
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}_steps_to_success.png"), dpi=200, bbox_inches="tight")

    plt.close("all")

def plot_granularity_comparison(metrics: list[dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    g = [m["granularity"] for m in metrics]
    success = [m["success_rate"] for m in metrics]
    wnorm = [m["mean_final_w_norm"] for m in metrics]
    steps = [m["mean_steps_to_success"] for m in metrics]

    # ---- Success rate vs granularidad ----
    plt.figure()
    plt.plot(g, success, marker="o")
    plt.xlabel("Granularidad")
    plt.ylabel("Success rate")
    plt.title("Success rate vs granularidad")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "granularity_vs_success.png"), dpi=200)

    # ---- ||ω|| final vs granularidad ----
    plt.figure()
    plt.plot(g, wnorm, marker="o")
    plt.xlabel("Granularidad")
    plt.ylabel("||ω|| final medio [rad/s]")
    plt.title("Norma final de velocidad angular vs granularidad")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "granularity_vs_w_norm.png"), dpi=200)

    # ---- Pasos al éxito vs granularidad ----
    plt.figure()
    plt.plot(g, steps, marker="o")
    plt.xlabel("Granularidad")
    plt.ylabel("Pasos medios al éxito")
    plt.title("Eficiencia vs granularidad")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "granularity_vs_steps.png"), dpi=200)

    plt.close("all")


# =========================
# Eval
# =========================
def eval_dqn(model_path, n_eval_episodes=50, seed=999, max_steps=400, granularity=40, time_step=1.0):
    eval_env = DummyVecEnv([make_env(
        max_steps=max_steps, granularity=granularity, time_step=time_step,
        seed=seed, log_dir=None
    )])

    model = DQN.load(model_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    eval_env.close()
    return mean_reward, std_reward


# =========================
# Train final with best params + auto-save best model
# =========================
def train_dqn_with_params(
    best_params: dict,
    total_timesteps: int = 300_000,
    seed: int = 123,
    best_dir: str = "best_model",
    log_dir: str = "logs_dqn",
    save_path_last: str = "models/dqn_last.zip",
    max_steps: int = 400,
    granularity: int = 40,
    time_step: float = 1.0,
    policy_kwargs: dict | None = None,
    device: str = "cuda",
    eval_freq: int = 20_000,
    n_eval_episodes: int = 5,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path_last), exist_ok=True)

    set_random_seed(seed)
    np.random.seed(seed)

    start_time = FIXED_START_TIME if USE_FIXED_START_TIME else None

    train_env = DummyVecEnv([make_env(
        max_steps=max_steps, granularity=granularity, time_step=time_step,
        seed=seed, log_dir=log_dir,
        start_time=start_time
    )])

    eval_env = DummyVecEnv([make_env(
        max_steps=max_steps, granularity=granularity, time_step=time_step,
        seed=seed + 1, log_dir=None,
        start_time=start_time
    )])


    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=15,
        min_evals=5,
        verbose=1
    )

    best_dir = os.path.join(best_dir)
    eval_dir = os.path.join(log_dir, "eval")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        callback_after_eval=stop_cb
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=float(best_params["learning_rate"]),
        gamma=float(best_params["gamma"]),
        batch_size=int(best_params["batch_size"]),
        buffer_size=int(best_params["buffer_size"]),
        learning_starts=10_000,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
    )

    print("SB3 device:", model.device)
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)

    model.save(save_path_last)
    best_model_path = os.path.join(best_dir, "best_model")

    eval_env.close()
    train_env.close()
    del model
    gc.collect()

    return best_model_path, save_path_last

def evaluate_for_granularity(
    model_path: str,
    granularity: int,
    n_eval_episodes: int,
    seed: int,
    max_steps: int,
    time_step: float,
):
    print(f"\n[GRAN] Evaluating granularity={granularity}")

    history = evaluate_with_history(
        model_path=model_path,
        n_eval_episodes=n_eval_episodes,
        seed=seed,
        max_steps=max_steps,
        granularity=granularity,
        time_step=time_step,
    )

    metrics = {
        "granularity": granularity,
        "success_rate": float(history["success_rate"]),
        "mean_reward": float(np.mean(history["rewards"])),
        "std_reward": float(np.std(history["rewards"])),
        "mean_final_w_norm": float(np.nanmean(history["final_w_norm"])),
        "std_final_w_norm": float(np.nanstd(history["final_w_norm"])),
        "mean_steps_to_success": (
            float(np.mean(history["steps_to_success"]))
            if len(history["steps_to_success"]) > 0 else np.nan
        ),
    }

    return metrics, history

def run_granularity_sweep(
    model_path: str,
    granularities: list[int],
    n_eval_episodes: int,
    seed: int,
    max_steps: int,
    time_step: float,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = []
    all_histories = {}

    for g in granularities:
        gran_seed = seed + g
        metrics, history = evaluate_for_granularity(
            model_path=model_path,
            granularity=g,
            n_eval_episodes=n_eval_episodes,
            seed=gran_seed,
            max_steps=max_steps,
            time_step=time_step,
        )

        all_metrics.append(metrics)
        all_histories[g] = history

        # plots individuales por granularidad
        plot_eval_history_scatter(
            history, save_dir=save_dir, prefix=f"gran_{g}"
        )

    # Guardar resultados numéricos
    summary = {
        "model_path": model_path,
        "base_seed": seed,
        "n_eval_episodes": n_eval_episodes,
        "max_steps": max_steps,
        "time_step": time_step,
        "granularities": granularities,
        "metrics": all_metrics,
    }

    results_path = os.path.join(save_dir, "granularity_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


    print(f"\n[GRAN] Results saved to {results_path}")

    return all_metrics, all_histories


# =========================
# Optuna: objective builder (closure)
# =========================
def make_objective(
    device: str,
    optuna_train_timesteps: int,
    optuna_eval_episodes: int,
    seed_train: int,
    seed_eval: int,
    max_steps: int,
    granularity: int,
    time_step: float,
    policy_kwargs: dict | None,
):
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])

        train_env = DummyVecEnv([make_env(
            max_steps=max_steps, granularity=granularity, time_step=time_step,
            seed=seed_train, log_dir=None
        )])

        eval_env = DummyVecEnv([make_env(
            max_steps=max_steps, granularity=granularity, time_step=time_step,
            seed=seed_eval, log_dir=None
        )])

        model = DQN(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=lr,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_starts=5_000,
            train_freq=4,
            target_update_interval=2_000,
            verbose=0,
            device=device,
        )

        model.learn(total_timesteps=optuna_train_timesteps)

        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=optuna_eval_episodes, deterministic=True
        )

        train_env.close()
        eval_env.close()
        del model
        gc.collect()

        return float(mean_reward)

    return objective


# =========================
# Optuna runner (n_trials as input)
# =========================
"""
def run_optuna(
    n_trials: int,
    device: str = "cuda",
    study_name: str = "dqn_cubesat",
    storage: str | None = None,          # ej: "sqlite:///optuna.db"
    load_if_exists: bool = True,
    pruner: optuna.pruners.BasePruner | None = None,
    optuna_train_timesteps: int = 50_000,
    optuna_eval_episodes: int = 20,
    seed_train: int = 0,
    seed_eval: int = 999,
    max_steps: int = 400,
    granularity: int = 40,
    time_step: float = 1.0,
    policy_kwargs: dict | None = None,
):
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        load_if_exists=load_if_exists,
        pruner=pruner,
        storage=storage
    )

    objective = make_objective(
        device=device,
        optuna_train_timesteps=optuna_train_timesteps,
        optuna_eval_episodes=optuna_eval_episodes,
        seed_train=seed_train,
        seed_eval=seed_eval,
        max_steps=max_steps,
        granularity=granularity,
        time_step=time_step,
        policy_kwargs=policy_kwargs,
    )

    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start

    return study, elapsed
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluate_success_rate(
    model_path: str,
    n_eval_episodes: int = 50,
    seed: int = 999,
    max_steps: int = 400,
    granularity: int = 40,
    time_step: float = 1.0,
):
    """
    Evalúa un modelo SB3 y retorna:
      - mean_reward, std_reward
      - success_rate (terminated True)
      - avg_steps_to_success (solo para episodios exitosos)
    """
    # Normaliza ruta (si viene con .zip)
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]

    eval_env = DummyVecEnv([make_env(
        max_steps=max_steps, granularity=granularity, time_step=time_step,
        seed=seed, log_dir=None
    )])

    model = DQN.load(model_path)

    episode_rewards = []
    success_flags = []
    steps_to_success = []

    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        done = [False]
        ep_reward = 0.0
        steps = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)

            ep_reward += float(reward[0])
            steps += 1

            # "done" en VecEnv corresponde a terminated OR truncated.
            # Para éxito usamos tu señal: terminated=True. En VecEnv, eso no viene directo,
            # así que lo inferimos desde info["success"] que tú retornas.
            if done[0]:
                info = infos[0]
                # print("infoooooo: ", info)
                success = bool(info.get("success", False))
                success_flags.append(success)

                if success:
                    steps_to_success.append(steps)

        episode_rewards.append(ep_reward)

    eval_env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    success_rate = float(np.mean(success_flags)) if len(success_flags) > 0 else 0.0

    avg_steps_success = float(np.mean(steps_to_success)) if len(steps_to_success) > 0 else float("nan")
    std_steps_success = float(np.std(steps_to_success)) if len(steps_to_success) > 0 else float("nan")

    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "avg_steps_on_success": avg_steps_success,
        "std_steps_on_success": std_steps_success,
        "n_eval_episodes": n_eval_episodes,
    }

    return metrics


def make_run_dirs(base_dir: str = "runs", exp_name: str = "dqn_cubesat", seed: int = 123):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, exp_name, f"seed_{seed}", run_id)

    paths = {
        "run_dir": run_dir,
        "log_dir": os.path.join(run_dir, "logs"),          # tensorboard + monitor
        "best_dir": os.path.join(run_dir, "best"),         # EvalCallback guarda best_model.zip aquí
        "models_dir": os.path.join(run_dir, "models"),     # last model, etc.
        "plots_dir": os.path.join(run_dir, "plots"),
        "results_path": os.path.join(run_dir, "dqn_results.json"),
    }

    for p in paths.values():
        # results_path es archivo, no carpeta
        if p.endswith(".json"):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        else:
            os.makedirs(p, exist_ok=True)

    return paths



def numpy_json_default(obj):
    # Arrays -> listas
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Escalares numpy -> escalares Python
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    # Si aparece algo raro, lanza el error original
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    USE_FIXED_START_TIME = True
    FIXED_START_TIME = datetime(2025, 1, 1)

    # ---- Switches / inputs ----
    RUN_OPTUNA = True

    # Optuna inputs
    N_TRIALS = 15
    OPTUNA_TRAIN_TIMESTEPS = 50_000
    OPTUNA_EVAL_EPISODES = 5

    # Final training inputs
    EVAL_FREQ = 20_000
    N_EVAL_EPISODES = 5
    
    FINAL_TIMESTEPS = 900_000
    FINAL_EVAL_EPISODES = 100

    # Env inputs
    SEED = 123
    MAX_STEPS = 400
    GRANULARITY = 40
    TIME_STEP = 0.1

    # I/O
    DEVICE = "cuda"  # o "auto"
    
    paths = make_run_dirs(base_dir="runs", exp_name="dqn_cubesat", seed=SEED)
    LOG_DIR = paths["log_dir"]
    BEST_DIR = paths["best_dir"]
    SAVE_LAST = os.path.join(paths["models_dir"], "dqn_last.zip")
    RESULTS_PATH = paths["results_path"]
    PLOTS_DIR = paths["plots_dir"]

    # Network size
    policy_kwargs = dict(net_arch=[256, 256])

    """
    # 0) Sanity check env
    print("\n[0] check_env...")
    tmp = CubeSatDetumblingEnv(render_mode=None, debug=False, plot_hist=False,
                              max_steps=MAX_STEPS, granularity=GRANULARITY, time_step=TIME_STEP)
    check_env(tmp, warn=True)
    tmp.close()
    print("[0] OK ✅\n")

    # 1) Optuna
    if RUN_OPTUNA:
        print("[1] Running Optuna...")
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study, elapsed = run_optuna(
            n_trials=N_TRIALS,
            device=DEVICE,
            study_name="dqn_cubesat",
            storage=None,  # si quieres persistir: "sqlite:///optuna.db"
            load_if_exists=True,
            pruner=pruner,
            optuna_train_timesteps=OPTUNA_TRAIN_TIMESTEPS,
            optuna_eval_episodes=OPTUNA_EVAL_EPISODES,
            seed_train=0,
            seed_eval=999,
            max_steps=MAX_STEPS,
            granularity=GRANULARITY,
            time_step=TIME_STEP,
            policy_kwargs=policy_kwargs,
        )
        best_params = study.best_params
        print(f"[1] Optuna done in {elapsed:.1f}s")
        print("[1] Best params:", best_params, "\n")
    else:
        best_params = {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 128, "buffer_size": 200_000}
        print("[1] Skipping Optuna. Using:", best_params, "\n")

    # 2) Final training with auto-best-save
    print("[2] Training final model (auto-best-save)...")
    best_model_path, last_model_path = train_dqn_with_params(
        best_params=best_params,
        total_timesteps=FINAL_TIMESTEPS,
        seed=SEED,
        log_dir=LOG_DIR,
        best_dir=BEST_DIR,
        save_path_last=SAVE_LAST,
        max_steps=MAX_STEPS,
        granularity=GRANULARITY,
        time_step=TIME_STEP,
        policy_kwargs=policy_kwargs,
        device=DEVICE,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    print(f"[2] Best model: {best_model_path}")
    print(f"[2] Last model: {last_model_path}\n")

    
    # 3) Evaluate best model
    print("[3] Evaluating best model (with success rate + plots)...")

    best_model_path = "/home/mapacheroja/apr-lab-2/20260107_063620/best/best_model.zip"

    print("Evaluating...")
    eval_metrics = evaluate_with_history(
        best_model_path,
        n_eval_episodes=FINAL_EVAL_EPISODES,
        seed=999,
        max_steps=MAX_STEPS,
        granularity=GRANULARITY,
        time_step=TIME_STEP
    )
    print("Evaluating done.")

    print(f"[3] Success rate: {eval_metrics['success_rate']*100:.2f}%")
    print(f"[3] Mean reward: {eval_metrics['rewards'].mean():.2f} ± {eval_metrics['rewards'].std():.2f}")
    print(f"[3] Mean final ||ω||: {np.nanmean(eval_metrics['final_w_norm']):.4f}")

    plot_eval_history_scatter(eval_metrics, save_dir=PLOTS_DIR, prefix="best_model_eval")
    # 4) Save summary
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "device": DEVICE,
        "optuna": {
            "run": RUN_OPTUNA,
            "n_trials": N_TRIALS,
            "train_timesteps": OPTUNA_TRAIN_TIMESTEPS,
            "eval_episodes": OPTUNA_EVAL_EPISODES
        },
        #"final_training": {
        #    "timesteps": FINAL_TIMESTEPS,
        #    "eval_freq": EVAL_FREQ,
        #    "n_eval_episodes_during_train": N_EVAL_EPISODES,
        #    "policy_kwargs": policy_kwargs,
        # },
        # "best_params": best_params,
        # "final_eval": eval_metrics,
        # "paths": {
        #    "best_model": best_model_path,
        #    "last_model": last_model_path,
        #    "log_dir": LOG_DIR,
        #},
    }

    model = DQN.load(best_model_path)
    print("Model loaded successfully.")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Gamma: {model.gamma}")
    print(f"Batch size: {model.batch_size}")
    print(f"Buffer size: {model.buffer_size}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=numpy_json_default)

    monitor_path = "/home/mapacheroja/apr-lab-2/20260107_063620/logs/monitor.csv"

    plot_training_monitor(
        monitor_csv_path=monitor_path,
        window=50,
        save_dir=PLOTS_DIR,
        prefix="training"
    )

    plot_final_eval_from_json(
        json_path="20260107_063620/dqn_results.json",
        save_dir=PLOTS_DIR,
        prefix="final_eval"
    ) 

    """ 
    best_model_path = "/home/mapacheroja/apr-lab-2/20260107_063620/best/best_model.zip"

    print("\n[4] Running granularity sweep...")

    GRANULARITIES = [20, 40, 60, 80]

    gran_metrics, gran_histories = run_granularity_sweep(
        model_path=best_model_path,
        granularities=GRANULARITIES,
        n_eval_episodes=100,      # puedes bajar a 30 si quieres rapidez
        seed=SEED,
        max_steps=MAX_STEPS,
        time_step=TIME_STEP,
        save_dir=os.path.join(PLOTS_DIR, "granularity"),
    )

    plot_granularity_comparison(
        gran_metrics,
        save_dir=os.path.join(PLOTS_DIR, "granularity")
    )

    print("[4] Granularity sweep done ✅")

    print(f"[5] Results saved to: {RESULTS_PATH}")
    print("Done ✅")
