"""
Deep Reinforcement Learning Training Pipeline for CubeSat Detumbling

This module implements a comprehensive training pipeline for Deep Q-Network (DQN) agents
designed to control CubeSat detumbling maneuvers. It provides functionality for:

- Environment creation and configuration with reproducible random seeds
- DQN model training with hyperparameter optimization using Optuna
- Model evaluation with detailed metrics and visualization
- Training monitoring and result plotting
- Multi-seed experiments for statistical analysis
- Granularity sweeps to assess model robustness across different environment resolutions

The module follows Stable-Baselines3 conventions and integrates with custom CubeSat
detumbling environments. All functions include comprehensive documentation and type hints
for maintainability and ease of use.

Example:
    >>> # Train a DQN model with default parameters
    >>> best_params = {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 128, "buffer_size": 200_000}
    >>> best_model_path, last_model_path = train_dqn_with_params(best_params)
    >>> 
    >>> # Evaluate the trained model
    >>> history = evaluate_with_history(best_model_path, n_eval_episodes=50)
    >>> print(f"Success rate: {history['success_rate']*100:.2f}%")
"""

import os
import json
import time
import gc
from datetime import datetime
import pandas as pd
import numpy as np
import optuna
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

def make_env(max_steps=400, granularity=40, time_step=0.1, seed=0, log_dir=None, start_time=None, reward_scaling=1.0):
    """
    Creates and initializes a new environment for the CubeSat detumbling problem.

    This function is used to wrap the `CubeSatDetumblingEnv` in a Monitor for logging, and
    to ensure reproducibility by setting a fixed seed for the environment. Additionally, it
    allows logging to be saved to a specified directory.

    Parameters:
    - max_steps (int): Maximum number of steps per episode.
    - granularity (int): The level of granularity for the environment.
    - time_step (float): The time step used in the simulation.
    - seed (int): The random seed for environment reproducibility.
    - log_dir (str or None): Directory to store logs, or None if no logging is required.
    - start_time (datetime or None): The fixed start time for the simulation, used for reproducibility.
    """
    def _init():

        # Initialize the CubeSat detumbling environment with the specified parameters
        env = CubeSatDetumblingEnv(
            render_mode=None,
            max_steps=max_steps,
            granularity=granularity,
            time_step=time_step,
            start_time=start_time,
            debug=False,
            plot_hist=False,
            reward_scaling=reward_scaling
        )

        env.reset(seed=seed)

        # Set up monitoring (logging) if a log directory is provided
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
    Evaluates the given model over multiple episodes and returns a dictionary of episode histories.
    
    This function loads a trained model from the specified path and evaluates it over a specified 
    number of episodes in the environment. It tracks various metrics for each episode, including 
    rewards, success rate, time taken, and other environment-specific information.
    
    The function is compatible with environments wrapped using `DummyVecEnv` for parallel evaluation.

    Parameters:
    - model_path (str): The path to the trained model file. Should be in `.zip` format.
    - n_eval_episodes (int): The number of episodes to evaluate the model.
    - seed (int): The random seed used to initialize the environment.
    - max_steps (int): The maximum number of steps per episode.
    - granularity (int): The granularity level for the environment's simulation.
    - time_step (float): The time step for the environment's simulation.

    Returns:
    - dict: A dictionary containing the following metrics for each episode:
        - "rewards" (list): The total reward accumulated in each episode.
        - "success" (list): A boolean indicating whether the episode was successful.
        - "lengths" (list): The number of steps taken in each episode.
        - "times" (list): The time taken for each episode.
        - "final_w_norm" (list): The final angular velocity norm at the end of each episode.
        - "steps_to_success" (list): The number of steps it took to achieve success (if any).
        - "n_eval_episodes" (int): The number of episodes evaluated.
        - "success_rate" (float): The proportion of successful episodes.
    """

    model_path = os.path.abspath(model_path)

    if not model_path.endswith(".zip") and os.path.isfile(model_path + ".zip"):
        model_path += ".zip"

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

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

    # Evaluate the model over the specified number of episodes
    for ep in range(n_eval_episodes):
        ep_start = time.time()
        obs = eval_env.reset()
        done = [False]
        ep_reward = 0.0
        steps = 0
        last_info = None

        # Run the episode until done
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)

            ep_reward += float(reward[0])
            steps += 1
            last_info = infos[0]  # info del único env

        ep_time = time.time() - ep_start
        print(f"[EVAL] Episode {ep+1}/{n_eval_episodes} "
              f"steps={steps} time={ep_time:.2f}s")
        
        # Extract specific information from the environment's monitoring data
        s = bool(last_info.get("success", False))
        w = float(last_info.get("angular_velocity_norm", np.nan))

        # Extract episode-specific info (length and time)
        ep_info = last_info.get("episode", {})
        l = int(ep_info.get("l", steps))
        t = float(ep_info.get("t", np.nan))

        # Store the metrics for this episode
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
    """
    Plots various evaluation metrics over episodes, including:
    - Reward per episode (scatter plot)
    - Cumulative success rate (line plot)
    - Final angular velocity norm (scatter plot)
    - Histogram of steps to success

    Parameters:
    - history (dict): A dictionary containing the evaluation history, which includes:
        - "rewards": List of rewards per episode.
        - "success": List of success flags (True/False) per episode.
        - "final_w_norm": List of final angular velocity norms per episode.
        - "steps_to_success": List of steps required to achieve success for successful episodes.
        - "n_eval_episodes": Total number of episodes evaluated.
    - save_dir (str or None): Directory where the plots will be saved. If None, plots will not be saved.
    - prefix (str): Prefix for the filenames of saved plots (default is "eval").
    """
    
    # Extract the data from the history dictionary
    rewards = history["rewards"]
    success = history["success"]
    wnorm = history["final_w_norm"]
    steps_to_success = history["steps_to_success"]
    n = history["n_eval_episodes"]

    episodes = np.arange(1, n + 1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Reward per episode (scatter plot)
    plt.figure()
    plt.scatter(episodes, rewards, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation: Reward per episode")
    plt.grid(True)

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_reward_scatter.png"), dpi=200, bbox_inches="tight")

    # 2. Cumulative success rate (line plot)
    plt.figure()
    success_cum = np.cumsum(success.astype(int)) / episodes
    plt.plot(episodes, success_cum)
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.title(f"Evaluation: Cumulative success rate (final={success_cum[-1]*100:.2f}%)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_success_rate.png"), dpi=200, bbox_inches="tight")

    # 3. Final angular velocity norm (scatter plot)
    plt.figure()
    plt.scatter(episodes, wnorm, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("||ω|| final [rad/s]")
    plt.title("Evaluation: Final angular velocity norm")
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_final_omega_norm_scatter.png"), dpi=200, bbox_inches="tight")

    # 4. Histogram of steps to success (only for successful episodes)
    if len(steps_to_success) > 0:
        plt.figure()
        plt.hist(steps_to_success, bins=15)
        plt.xlabel("Steps to success")
        plt.ylabel("Frequency")
        plt.title("Evaluation: Distribution of steps to success")
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{prefix}_steps_to_success_hist.png"), dpi=200, bbox_inches="tight")

    # Close all plots to free memory
    plt.close("all")


def plot_training_monitor(
    monitor_csv_path: str,
    window: int = 50,
    save_dir: str | None = None,
    prefix: str = "training",
):
    """
    Reads a monitor.csv file from Stable-Baselines3 and plots training statistics:
    1) Reward per episode + moving average of rewards.
    2) Episode length (number of steps per episode).
    3) Reward vs. training time.

    Parameters:
    -----------
    monitor_csv_path : str
        The path to the monitor.csv file generated during training. This file contains episode rewards,
        lengths, and times.
    window : int
        The window size used for calculating the moving average of rewards. Default is 50.
    save_dir : str or None
        The directory where the plots will be saved. If None, plots will only be shown and not saved.
    prefix : str
        The prefix used for the filenames of saved plots. Default is "training".
    """

    if not os.path.isfile(monitor_csv_path):
        raise FileNotFoundError(f"No existe: {monitor_csv_path}")

    # Read the CSV file, ignoring comments (lines starting with '#')
    df = pd.read_csv(
        monitor_csv_path,
        comment="#"
    )

    # Ensure that the necessary columns are present in the CSV file
    assert {"r", "l", "t"}.issubset(df.columns), df.columns

    # Extract reward, episode length, and time data from the CSV file
    rewards = df["r"].to_numpy()
    lengths = df["l"].to_numpy()
    times = df["t"].to_numpy()

    episodes = np.arange(1, len(rewards) + 1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Reward per episode with moving average
    plt.figure()
    plt.plot(episodes, rewards, alpha=0.4, label="Reward per episode")

    # Calculate and plot the moving average of rewards
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            np.arange(window, len(rewards) + 1),
            ma,
            linewidth=2,
            label=f"Moving average ({window})"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training: Reward per episode")
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified directory, if provided
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_reward.png"),
            dpi=200,
            bbox_inches="tight"
        )

    # 2. Episode length
    plt.figure()
    plt.plot(episodes, lengths)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Training: Episode length")
    plt.grid(True)

    # Save the plot
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_episode_length.png"),
            dpi=200,
            bbox_inches="tight"
        )

    # 3. Reward vs time
    plt.figure()
    plt.plot(times / 60.0, rewards, alpha=0.5)
    plt.xlabel("Training time [min]")
    plt.ylabel("Reward")
    plt.title("Training: Reward vs time")
    plt.grid(True)

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"{prefix}_reward_vs_time.png"),
            dpi=200,
            bbox_inches="tight"
        )

    # Close all plots to free up memory
    plt.close("all")

def plot_final_eval_from_json(
    json_path: str,
    save_dir: str | None = None,
    prefix: str = "final_eval"
):
    """
    Generates plots from the 'final_eval' field in a DQN results JSON file.

    This function reads a JSON file containing the evaluation results of a trained model 
    and generates several plots, including:
    - Reward per episode
    - Cumulative success rate
    - Final angular velocity norm
    - Histogram of steps to success

    Parameters:
    -----------
    json_path : str
        Path to the JSON file containing the final evaluation data. The file must have a field
        called 'final_eval' with the necessary data.
    save_dir : str or None
        Directory where the plots will be saved. If None, the plots will only be displayed, not saved.
    prefix : str
        Prefix used for the filenames of the saved plots. Default is "final_eval".
    """

    # Read the JSON file containing evaluation results
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the 'final_eval' data from the JSON
    eval_data = data["final_eval"]

    # Convert the data into numpy arrays for easier manipulation
    rewards = np.array(eval_data["rewards"])
    success = np.array(eval_data["success"], dtype=int)
    lengths = np.array(eval_data["lengths"])
    wnorm = np.array(eval_data["final_w_norm"])
    steps_success = np.array(eval_data["steps_to_success"])
    n = eval_data["n_eval_episodes"]

    episodes = np.arange(1, n + 1)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Reward per episode
    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Final Evaluation: Reward per Episode")
    plt.grid(True)

    # Save the plot
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_reward.png"), dpi=200, bbox_inches="tight")

    # 2. Cumulative success rate
    plt.figure()
    success_cum = np.cumsum(success) / episodes
    plt.plot(episodes, success_cum)
    plt.ylim(0, 1.05)
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.title(f"Final Evaluation: Cumulative Success Rate ({success_cum[-1]*100:.1f}%)")
    plt.grid(True)

    # Save the plot
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_success_rate.png"), dpi=200, bbox_inches="tight")

    # 3. Final angular velocity norm
    plt.figure()
    plt.plot(episodes, wnorm)
    plt.xlabel("Episode")
    plt.ylabel("||ω|| final [rad/s]")
    plt.title("Final Evaluation: Angular Velocity Norm")
    plt.grid(True)

    # Save the plot
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_final_omega_norm.png"), dpi=200, bbox_inches="tight")

    # 4. Histogram of steps to success
    if len(steps_success) > 0:
        plt.figure()
        plt.hist(steps_success, bins=15)
        plt.xlabel("Steps to Success")
        plt.ylabel("Frequency")
        plt.title("Final Evaluation: Distribution of Steps to Success")
        plt.grid(True)

    # Save the plot
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_steps_to_success.png"), dpi=200, bbox_inches="tight")

    # Close all plots to free up memory
    plt.close("all")

def plot_granularity_comparison(metrics: list[dict], save_dir: str):
    """
    Plots performance metrics (success rate, final angular velocity norm, steps to success) 
    as a function of granularity.

    This function generates three plots:
    1) Success rate vs. granularity
    2) Final angular velocity norm vs. granularity
    3) Steps to success vs. granularity

    The plots are saved in the specified directory.

    Parameters:
    -----------
    metrics : list of dicts
        A list of dictionaries where each dictionary contains performance metrics for a specific granularity.
        Each dictionary should have the following keys:
        - "granularity" (int): The granularity value for the environment.
        - "success_rate" (float): The success rate achieved at this granularity.
        - "mean_final_w_norm" (float): The mean final angular velocity norm.
        - "mean_steps_to_success" (float): The mean number of steps to success.
    save_dir : str
        The directory where the generated plots will be saved. If the directory does not exist, it will be created.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract the granularity values and corresponding metrics
    g = [m["granularity"] for m in metrics]
    success = [m["success_rate"] for m in metrics]
    wnorm = [m["mean_final_w_norm"] for m in metrics]
    steps = [m["mean_steps_to_success"] for m in metrics]

    # Success rate vs granularity 
    plt.figure()
    plt.plot(g, success, marker="o")
    plt.xlabel("Granularity")
    plt.ylabel("Success rate")
    plt.title("Success rate vs granularity")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "granularity_vs_success.png"), dpi=200)

    # Final vs granularity
    plt.figure()
    plt.plot(g, wnorm, marker="o")
    plt.xlabel("Granularity")
    plt.ylabel("||ω|| final mean [rad/s]")
    plt.title("Final Angular Velocity Norm vs Granularity")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "granularity_vs_w_norm.png"), dpi=200)

    # Steps to success vs granularity 
    plt.figure()
    plt.plot(g, steps, marker="o")
    plt.xlabel("Granularity")
    plt.ylabel("Mean Steps to Success")
    plt.title("Steps to Success vs Granularity")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "granularity_vs_steps.png"), dpi=200)

    # Close all plots to free up memory
    plt.close("all")

def eval_dqn(
    model_path: str,
    n_eval_episodes: int = 50,
    seed: int = 999,
    max_steps: int = 400,
    granularity: int = 40,
    time_step: float = 1.0,
) -> tuple[float, float]:
    """
    Evaluates a trained DQN model and returns mean and standard deviation of rewards.

    This function loads a pre-trained DQN model and evaluates its performance over
    a specified number of episodes using the Stable-Baselines3 evaluate_policy function.
    The evaluation is deterministic, meaning the model will always select the same
    actions for the same states.

    Parameters:
    -----------
    model_path : str
        Path to the trained DQN model file (without .zip extension).
    n_eval_episodes : int, default=50
        Number of episodes to run for evaluation.
    seed : int, default=999
        Random seed for environment initialization to ensure reproducible evaluation.
    max_steps : int, default=400
        Maximum number of steps per episode in the evaluation environment.
    granularity : int, default=40
        Granularity level for the environment simulation.
    time_step : float, default=1.0
        Time step duration for the environment simulation.

    Returns:
    --------
    tuple[float, float]
        A tuple containing:
        - mean_reward: Mean reward across all evaluation episodes
        - std_reward: Standard deviation of rewards across all evaluation episodes
    """
    # Create evaluation environment with specified parameters
    eval_env = DummyVecEnv([make_env(
        max_steps=max_steps, granularity=granularity, time_step=time_step,
        seed=seed, log_dir=None
    )])

    # Load the trained DQN model
    model = DQN.load(model_path)

    # Evaluate the model using Stable-Baselines3's evaluate_policy function
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    # Clean up resources
    eval_env.close()
    return mean_reward, std_reward

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
) -> tuple[str, str]:
    """
    Trains a Deep Q-Network (DQN) agent with specified hyperparameters and evaluation callbacks.
    
    This function creates and trains a DQN model using the Stable-Baselines3 library with
    comprehensive evaluation and early stopping mechanisms. It sets up training and evaluation
    environments, configures callbacks for model evaluation and early stopping, and saves
    both the best and final model states.
    
    Parameters:
    -----------
    best_params : dict
        Dictionary containing optimized hyperparameters for the DQN model. Expected keys:
        - learning_rate: Learning rate for the optimizer
        - gamma: Discount factor for future rewards
        - batch_size: Number of experiences sampled from replay buffer
        - buffer_size: Size of the experience replay buffer
    total_timesteps : int, default=300_000
        Total number of timesteps to train the model.
    seed : int, default=123
        Random seed for reproducible training across environments and numpy.
    best_dir : str, default="best_model"
        Directory path where the best performing model will be saved.
    log_dir : str, default="logs_dqn"
        Base directory for TensorBoard logs and evaluation metrics.
    save_path_last : str, default="models/dqn_last.zip"
        File path where the final model (after training completion) will be saved.
    max_steps : int, default=400
        Maximum number of steps per episode in the environment.
    granularity : int, default=40
        Granularity level for environment simulation discretization.
    time_step : float, default=1.0
        Time step duration for the environment simulation.
    policy_kwargs : dict | None, default=None
        Additional keyword arguments for the policy network architecture.
    device : str, default="cuda"
        Computing device for model training ("cuda" or "cpu").
    eval_freq : int, default=20_000
        Frequency of evaluation during training (in timesteps).
    n_eval_episodes : int, default=5
        Number of episodes to run during each evaluation phase.
    
    Returns:
    --------
    tuple[str, str]
        A tuple containing:
        - best_model_path: Path to the best performing model saved during training
        - save_path_last: Path to the final model after training completion
    
    Notes:
    ------
    - Uses early stopping with StopTrainingOnNoModelImprovement to prevent overtraining
    - Implements EvalCallback for periodic model evaluation during training
    - Automatically creates necessary directories if they don't exist
    - Properly cleans up resources (environments and model) after training
    """
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

def train_dqn_with_multiple_trials(
    best_params: dict,
    total_timesteps: int = 300_000,
    n_trials: int = 5,
    seed: int = 123,
    log_dir: str = "logs_dqn",
    best_model_dir: str = "best_model",
    save_last_model_path: str = "models/dqn_last.zip",
    max_steps: int = 400,
    granularity: int = 40,
    time_step: float = 1.0,
    eval_freq: int = 20_000,
    n_eval_episodes: int = 5,
    policy_kwargs: dict = None,
    device: str = "cuda",
):
    """
    Performs multiple independent training trials with the same hyperparameters for statistical analysis.
    
    This function runs several complete training and evaluation cycles using identical hyperparameters
    but different random seeds to assess the consistency and reliability of the training process.
    Each trial includes a full training phase followed by evaluation, with results aggregated to provide
    statistical measures of performance.
    
    Parameters:
    -----------
    best_params : dict
        Dictionary containing optimized hyperparameters for the DQN model. Expected keys:
        - learning_rate: Learning rate for the optimizer
        - gamma: Discount factor for future rewards
        - batch_size: Number of experiences sampled from replay buffer
        - buffer_size: Size of the experience replay buffer
    total_timesteps : int, default=300_000
        Total number of timesteps to train each model in each trial.
    n_trials : int, default=5
        Number of independent training trials to execute.
    seed : int, default=123
        Base random seed. Each trial uses seed + trial_index for unique seeds.
    log_dir : str, default="logs_dqn"
        Base directory for TensorBoard logs and evaluation metrics.
    best_model_dir : str, default="best_model"
        Directory path where the best performing models will be saved.
    save_last_model_path : str, default="models/dqn_last.zip"
        File path where the final model after each trial will be saved.
    max_steps : int, default=400
        Maximum number of steps per episode in the environment.
    granularity : int, default=40
        Granularity level for environment simulation discretization.
    time_step : float, default=1.0
        Time step duration for the environment simulation.
    eval_freq : int, default=20_000
        Frequency of evaluation during training (in timesteps).
    n_eval_episodes : int, default=5
        Number of episodes to run during each evaluation phase.
    policy_kwargs : dict, default=None
        Additional keyword arguments for the policy network architecture.
    device : str, default="cuda"
        Computing device for model training ("cuda" or "cpu").
    
    Returns:
    --------
    tuple[float, float]
        A tuple containing:
        - avg_reward: Mean reward across all trials
        - median_reward: Median reward across all trials
    
    Notes:
    ------
    - Each trial uses a different random seed (seed + trial_index) to ensure variability
    - Evaluation uses a fixed seed (999) for consistent comparison across trials
    - Results are aggregated to provide statistical measures of performance consistency
    - Only the evaluation rewards are used for statistical analysis, not training rewards
    """
    # Lista para guardar los resultados de cada trial
    trial_rewards = []

    # Ejecutar varios trials (repeticiones)
    for trial in range(n_trials):
        print(f"\nRunning trial {trial+1}/{n_trials}")

        # Entrenar el modelo para este trial con los parámetros actuales
        best_model_path, _ = train_dqn_with_params(
            best_params=best_params,
            total_timesteps=total_timesteps,
            seed=seed + trial,  # Aseguramos que cada trial tiene una semilla diferente
            log_dir=log_dir,
            best_dir=best_model_dir,
            save_path_last=save_last_model_path,
            max_steps=max_steps,
            granularity=granularity,
            time_step=time_step,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            policy_kwargs=policy_kwargs,
            device=device,
        )

        # Evaluar el modelo después de entrenarlo
        print(f"Evaluating model from trial {trial + 1}...")
        history = evaluate_with_history(
            best_model_path,
            n_eval_episodes=n_eval_episodes,
            seed=999,  # Usamos un seed fijo para la evaluación
            max_steps=max_steps,
            granularity=granularity,
            time_step=time_step,
        )

        # Guardar las recompensas de este trial para calcular el promedio
        trial_rewards.append(np.mean(history["rewards"]))  # Promedio de las recompensas por episodio

    # Promediar los resultados de todos los trials
    avg_reward = np.mean(trial_rewards)
    median_reward = np.median(trial_rewards)

    print(f"\nAverage reward over {n_trials} trials: {avg_reward:.2f}")
    print(f"Median reward over {n_trials} trials: {median_reward:.2f}")

    return avg_reward, median_reward

def run_experiment_seeds(
    n_seeds=5, 
    total_timesteps=300_000, 
    eval_episodes=50, 
    seed_base=123,
    max_steps=400, 
    granularity=40, 
    time_step=0.1, 
    log_dir="logs_dqn",
    model_save_dir="models",
    eval_freq=20_000
):
    """
    Runs multiple training experiments with different random seeds for statistical analysis.
    This function trains and evaluates DQN models across multiple random seeds to assess
    the statistical robustness of the training process. Each seed gets its own directory
    structure, and results are aggregated into a comprehensive DataFrame for analysis.
    Parameters:
    -----------
    n_seeds : int, default=5
        Number of different random seeds to run experiments with.
    total_timesteps : int, default=300_000
        Number of timesteps to train each model.
    eval_episodes : int, default=50
        Number of episodes to use for final evaluation of each trained model.
    seed_base : int, default=123
        Base seed value. Actual seeds used will be seed_base, seed_base+1, ..., seed_base+n_seeds-1.
    max_steps : int, default=400
        Maximum number of steps per episode in the environment.
    granularity : int, default=40
        Granularity level for the environment simulation.
    time_step : float, default=0.1
        Time step duration for the environment simulation.
    log_dir : str, default="logs_dqn"
        Base directory for logs (actual logs go in seed-specific subdirectories).
    model_save_dir : str, default="models"
        Base directory for saving models (actual models go in seed-specific subdirectories).
    eval_freq : int, default=20_000
        Frequency of evaluation during training (in timesteps).
    Returns:
    --------
    pd.DataFrame
        DataFrame containing aggregated results from all seeds with columns:
        - seed: The random seed used
        - mean_reward: Mean reward across evaluation episodes
        - std_reward: Standard deviation of rewards
        - median_reward: Median reward across evaluation episodes
        - success_rate: Proportion of successful episodes
        - mean_final_w_norm: Mean final angular velocity norm
        - std_final_w_norm: Standard deviation of final angular velocity norm
    """
    # Run experiments for each seed
    all_metrics = []

    for seed in range(seed_base, seed_base + n_seeds):
        print(f"Running experiment with seed {seed}")

        # Create directories for this seed
        paths = make_run_dirs(base_dir="runs", exp_name="dqn_cubesat", seed=seed)
        log_dir = paths["log_dir"]
        best_model_dir = paths["best_dir"]
        save_last = os.path.join(paths["models_dir"], f"dqn_last_seed{seed}.zip")

        # Set random seeds
        set_random_seed(seed)
        np.random.seed(seed)

        # Train model
        print("[Training] Starting training...")
        best_model_path, last_model_path = train_dqn_with_params(
            best_params={
                "learning_rate": 1e-4, 
                "gamma": 0.99, 
                "batch_size": 128, 
                "buffer_size": 200_000
            },
            total_timesteps=total_timesteps,
            seed=seed,
            log_dir=log_dir,
            best_dir=best_model_dir,
            save_path_last=save_last,
            max_steps=max_steps,
            granularity=granularity,
            time_step=time_step,
            eval_freq=eval_freq,
        )

        # Evaluate the model
        print(f"[Evaluation] Evaluating model {best_model_path} for seed {seed}...")
        history = evaluate_with_history(
            best_model_path,
            n_eval_episodes=eval_episodes,
            seed=999,  # Evaluamos con un seed fijo
            max_steps=max_steps,
            granularity=granularity,
            time_step=time_step
        )

        # Save metrics for this seed
        metrics = {
            "seed": seed,
            "mean_reward": np.mean(history["rewards"]),
            "std_reward": np.std(history["rewards"]),
            "median_reward": np.median(history["rewards"]),
            "success_rate": history["success_rate"],
            "mean_final_w_norm": np.nanmean(history["final_w_norm"]),
            "std_final_w_norm": np.nanstd(history["final_w_norm"]),
        }
        all_metrics.append(metrics)

        # Print metrics for each seed
        print(f"Seed {seed} - Success rate: {metrics['success_rate']*100:.2f}%")
        print(f"Seed {seed} - Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Seed {seed} - Final ||ω||: {metrics['mean_final_w_norm']:.4f} ± {metrics['std_final_w_norm']:.4f}")

    # Create DataFrame with all results
    df = pd.DataFrame(all_metrics)

    # Save final summary
    df.to_csv("experiment_results.csv", index=False)
    print(f"Results saved to experiment_results.csv")
    
    return df

def evaluate_for_granularity(
    model_path: str,
    granularity: int,
    n_eval_episodes: int,
    seed: int,
    max_steps: int,
    time_step: float,
):
    """
    Evaluates a trained model at a specific granularity level.
    This function assesses model performance at a particular environment granularity
    by running multiple evaluation episodes and computing comprehensive metrics.
    It's designed to be used as part of granularity sweep analysis.
    Parameters:
    -----------
    model_path : str
        Path to the trained model file.
    granularity : int
        Granularity level at which to evaluate the model.
    n_eval_episodes : int
        Number of episodes to run for evaluation.
    seed : int
        Random seed for reproducible evaluation.
    max_steps : int
        Maximum number of steps per episode.
    time_step : float
        Time step duration for the environment simulation.
    Returns:
    --------
    tuple[dict, dict]
        A tuple containing:
        - metrics: Dictionary with performance metrics including success rate,
          mean reward, final angular velocity norm, and steps to success
        - history: Full evaluation history with detailed episode data
    """
    print(f"\n[GRAN] Evaluating granularity={granularity}")

    # Run evaluation with the specified granularity
    history = evaluate_with_history(
        model_path=model_path,
        n_eval_episodes=n_eval_episodes,
        seed=seed,
        max_steps=max_steps,
        granularity=granularity,
        time_step=time_step,
    )

    # Compute metrics
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
    """
    Performs a comprehensive granularity sweep analysis for a trained model.
    This function evaluates a trained model across multiple granularity levels
    to assess how environment resolution affects performance. It generates
    individual plots for each granularity and saves a comprehensive summary
    of all results.
    Parameters:
    -----------
    model_path : str
        Path to the trained model to evaluate.
    granularities : list[int]
        List of granularity values to test.
    n_eval_episodes : int
        Number of episodes to run for each granularity evaluation.
    seed : int
        Base seed for reproducible evaluation (actual seeds will be seed + granularity).
    max_steps : int
        Maximum number of steps per episode.
    time_step : float
        Time step duration for the environment simulation.
    save_dir : str
        Directory where plots and results will be saved.
    Returns:
    --------
    tuple[list[dict], dict[int, dict]]
        A tuple containing:
        - all_metrics: List of metric dictionaries, one for each granularity tested
        - all_histories: Dictionary mapping granularity values to full evaluation histories
    """
    # Create output directory for results
    os.makedirs(save_dir, exist_ok=True)

    # Store results for all granularities
    all_metrics = []
    all_histories = {}

    # Evaluate model for each granularity
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

        # Individual plots by granularity
        plot_eval_history_scatter(
            history, save_dir=save_dir, prefix=f"gran_{g}"
        )

    # Save numerical results
    summary = {
        "model_path": model_path,
        "base_seed": seed,
        "n_eval_episodes": n_eval_episodes,
        "max_steps": max_steps,
        "time_step": time_step,
        "granularities": granularities,
        "metrics": all_metrics,
    }
    
    # Save results to JSON file for external analysis
    results_path = os.path.join(save_dir, "granularity_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[GRAN] Results saved to {results_path}")

    return all_metrics, all_histories

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
    """
    Creates an Optuna objective function for DQN hyperparameter optimization.
    This function returns a closure that can be used with Optuna to optimize DQN
    hyperparameters. The objective function trains a DQN model with suggested
    hyperparameters and returns the mean evaluation reward as the optimization target.
    Parameters:
    -----------
    device : str
        Device to use for training ("cuda", "cpu", or "auto").
    optuna_train_timesteps : int
        Number of timesteps to train each model during hyperparameter search.
    optuna_eval_episodes : int
        Number of episodes to use for evaluation during hyperparameter search.
    seed_train : int
        Random seed for training environment.
    seed_eval : int
        Random seed for evaluation environment.
    max_steps : int
        Maximum number of steps per episode.
    granularity : int
        Granularity level for the environment simulation.
    time_step : float
        Time step duration for the environment simulation.
    policy_kwargs : dict or None
        Additional keyword arguments for the policy network architecture.
    Returns:
    --------
    callable
        An objective function that takes an Optuna trial object and returns the
        mean evaluation reward for the suggested hyperparameters.
    """
    def objective(trial):
        """
        Optuna objective function for DQN hyperparameter optimization.
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object used to suggest hyperparameters.
        Returns:
        --------
        float
            Mean reward across evaluation episodes (to be maximized).
        """

        # Suggest hyperparameters using Optuna's sampling methods
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])
        neurons = trial.suggest_categorical("neurons", [64, 128, 256])
        activation_function = trial.suggest_categorical("activation_function", ["relu", "tanh"])
        reward_scaling = trial.suggest_categorical("reward_scaling", [0.1, 1.0, 10.0])
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        target_update_interval = trial.suggest_int("target_update_interval", 1000, 10000)
        train_freq = trial.suggest_int("train_freq", 1, 10)

        policy_kwargs = dict(
            net_arch=[neurons, neurons],
            activation_fn=activation_function,
        )

        # Create and configure DQN model with suggested hyperparameters
        train_env = DummyVecEnv([make_env(
            max_steps=max_steps, granularity=granularity, time_step=time_step,
            seed=seed_train, log_dir=None, reward_scaling=reward_scaling
        )])

        eval_env = DummyVecEnv([make_env(
            max_steps=max_steps, granularity=granularity, time_step=time_step,
            seed=seed_eval, log_dir=None, reward_scaling=reward_scaling
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
            train_freq=train_freq,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            verbose=0,
            device=device,
        )

        # Train the model with the suggested hyperparameters
        model.learn(total_timesteps=optuna_train_timesteps)

        # Evaluate the trained model
        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=optuna_eval_episodes, deterministic=True
        )

        # Clean up resources to prevent memory leaks
        train_env.close()
        eval_env.close()
        del model
        gc.collect()

        return float(mean_reward)

    return objective

def run_optuna(
    n_trials: int,
    device: str = "cuda",
    study_name: str = "dqn_cubesat",
    storage: str | None = None,
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
    """
    Executes hyperparameter optimization using Optuna for DQN model training.
    
    This function orchestrates a comprehensive hyperparameter optimization study using Optuna
    to find optimal DQN hyperparameters. It creates or loads an optimization study, configures
    the objective function with the specified environment and training parameters, and runs
    the optimization for a specified number of trials. The study can be persisted to storage
    for later analysis and continuation.
    
    Parameters:
    -----------
    n_trials : int
        Number of optimization trials to execute in the study.
    device : str, default="cuda"
        Computing device for model training ("cuda" or "cpu").
    study_name : str, default="dqn_cubesat"
        Name of the Optuna study for identification and storage.
    storage : str | None, default=None
        Database storage URL for persisting the study (e.g., "sqlite:///optuna.db").
        If None, study is stored in memory only.
    load_if_exists : bool, default=True
        Whether to load existing study if found in storage.
    pruner : optuna.pruners.BasePruner | None, default=None
        Pruning strategy for early termination of unpromising trials.
        If None, uses MedianPruner with default parameters.
    optuna_train_timesteps : int, default=50_000
        Number of training timesteps for each optimization trial.
    optuna_eval_episodes : int, default=20
        Number of evaluation episodes for each optimization trial.
    seed_train : int, default=0
        Random seed for training during optimization trials.
    seed_eval : int, default=999
        Random seed for evaluation during optimization trials.
    max_steps : int, default=400
        Maximum number of steps per episode in the environment.
    granularity : int, default=40
        Granularity level for environment simulation discretization.
    time_step : float, default=1.0
        Time step duration for the environment simulation.
    policy_kwargs : dict | None, default=None
        Additional keyword arguments for the policy network architecture.
    
    Returns:
    --------
    tuple[optuna.Study, float]
        A tuple containing:
        - study: The completed Optuna study object with optimization results
        - elapsed: Total time taken for the optimization in seconds
    
    Notes:
    ------
    - Default pruner is MedianPruner with 5 startup trials and 1 warmup step
    - Study direction is "maximize" (optimizing for higher rewards)
    - Each trial uses shortened training (50k timesteps) for faster optimization
    - Results can be analyzed and visualized using Optuna's built-in tools
    - Study persistence allows for continuation and analysis across sessions
    """
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

def evaluate_success_rate(
    model_path: str,
    n_eval_episodes: int = 50,
    seed: int = 999,
    max_steps: int = 400,
    granularity: int = 40,
    time_step: float = 1.0,
):
    """
    Evaluates a Stable-Baselines3 model and returns comprehensive success metrics.
    This function loads a trained model and evaluates it over multiple episodes,
    tracking not just rewards but also success rates and steps to success for
    successful episodes. It provides more detailed metrics than standard evaluation.
    Parameters:
    -----------
    model_path : str
        Path to the trained model file (can include or exclude .zip extension).
    n_eval_episodes : int, default=50
        Number of episodes to run for evaluation.
    seed : int, default=999
        Random seed for reproducible evaluation.
    max_steps : int, default=400
        Maximum number of steps per episode.
    granularity : int, default=40
        Granularity level for the environment simulation.
    time_step : float, default=1.0
        Time step duration for the environment simulation.
    Returns:
    --------
    dict
        Dictionary containing comprehensive evaluation metrics:
        - mean_reward: Mean reward across all episodes
        - std_reward: Standard deviation of rewards
        - success_rate: Proportion of successful episodes
        - avg_steps_on_success: Mean steps to success for successful episodes only
        - std_steps_on_success: Standard deviation of steps to success
        - n_eval_episodes: Number of episodes evaluated
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
    """
    Creates a structured directory hierarchy for experiment runs.
    This function generates a timestamped directory structure for organizing
    experiment results, with separate subdirectories for logs, models, plots,
    and other outputs. The structure follows the pattern: base_dir/exp_name/seed_X/timestamp/
    Parameters:
    -----------
    base_dir : str, default="runs"
        Base directory where all experiment runs will be stored.
    exp_name : str, default="dqn_cubesat"
        Name of the experiment (used as a subdirectory).
    seed : int, default=123
        Random seed for the experiment (used in the directory structure).
    Returns:
    --------
    dict
        Dictionary containing paths for all experiment directories and files:
        - run_dir: Root directory for this specific run
        - log_dir: Directory for training logs and TensorBoard data
        - best_dir: Directory where best models are saved by EvalCallback
        - models_dir: Directory for final models and checkpoints
        - plots_dir: Directory for generated plots and visualizations
        - results_path: Path to the main results JSON file
    """

    # Generate unique timestamp for the run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, exp_name, f"seed_{seed}", run_id)

    # Define all necessary paths for the experiment
    paths = {
        "run_dir": run_dir,
        "log_dir": os.path.join(run_dir, "logs"),          # tensorboard + monitor
        "best_dir": os.path.join(run_dir, "best"),         # EvalCallback guarda best_model.zip aquí
        "models_dir": os.path.join(run_dir, "models"),     # last model, etc.
        "plots_dir": os.path.join(run_dir, "plots"),
        "results_path": os.path.join(run_dir, "dqn_results.json"),
    }

    # Create all directories (handle file paths separately)
    for p in paths.values():
        # results_path is a file, not a directory
        if p.endswith(".json"):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        else:
            os.makedirs(p, exist_ok=True)

    return paths

def numpy_json_default(obj):
    """
    Converts NumPy objects to JSON-serializable Python types.
    
    This function serves as a default serializer for JSON encoding when handling
    NumPy objects that are not natively JSON serializable. It handles NumPy arrays
    and scalar types by converting them to their equivalent Python built-in types.
    This is commonly used as the default parameter in json.dumps() when saving
    experimental results or model metrics that contain NumPy data.
    
    Parameters:
    -----------
    obj : Any
        The object to be converted to a JSON-serializable type. Expected types:
        - numpy.ndarray: Converted to Python list
        - numpy.float32, numpy.float64: Converted to Python float
        - numpy.int32, numpy.int64: Converted to Python int
        - numpy.bool_: Converted to Python bool
    
    Returns:
    --------
    Any
        JSON-serializable equivalent of the input object:
        - list for NumPy arrays
        - float for NumPy float types
        - int for NumPy integer types
        - bool for NumPy boolean types
    
    Raises:
    -------
    TypeError
        If the object type is not supported for JSON serialization.
        This includes any non-NumPy objects that don't have built-in JSON support.
    
    Notes:
    ------
    - This function is essential for saving experiment results, metrics, and
      hyperparameters that contain NumPy data structures to JSON files
    - Common usage pattern: json.dumps(data, default=numpy_json_default)
    - Preserves data structure integrity while ensuring JSON compatibility
    - Only handles NumPy types; other custom objects will raise TypeError
    """

    # Arrays to lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Numpy scalars to Python scalars
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    # Numpy integers to Python integers
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    # Numpy booleans to Python booleans
    if isinstance(obj, (np.bool_)):
        return bool(obj)

    # If something weird appears, raise the original error
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

if __name__ == "__main__":
    # Fixed start time
    USE_FIXED_START_TIME = True
    FIXED_START_TIME = datetime(2025, 1, 1)

    # Optuna
    RUN_OPTUNA = True
    N_TRIALS = 15
    OPTUNA_TRAIN_TIMESTEPS = 50_000
    OPTUNA_EVAL_EPISODES = 5

    # Final training inputs
    EVAL_FREQ = 20_000
    N_EVAL_EPISODES = 5
    FINAL_TIMESTEPS = 900_000
    FINAL_EVAL_EPISODES = 100

    # Environment inputs
    SEED = 123
    MAX_STEPS = 400
    GRANULARITY = 40
    TIME_STEP = 0.1

    # I/O
    DEVICE = "cuda"  # o "auto"

    # Create experiment directories
    paths = make_run_dirs(base_dir="runs", exp_name="dqn_cubesat", seed=SEED)
    LOG_DIR = paths["log_dir"]
    BEST_DIR = paths["best_dir"]
    SAVE_LAST = os.path.join(paths["models_dir"], "dqn_last.zip")
    RESULTS_PATH = paths["results_path"]
    PLOTS_DIR = paths["plots_dir"]

    # Define the experiment parameters
    N_SEEDS = 5
    TOTAL_TIMESTEPS = 300_000
    EVAL_EPISODES = 50
    SEED_BASE = 123  # Base seed

    # After defining everything, we run the experiment
    experiment_df = run_experiment_seeds(
        n_seeds=N_SEEDS,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_episodes=EVAL_EPISODES,
        seed_base=SEED_BASE
    )

    # Save experiment results
    filename = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Save DataFrame as CSV and JSON
    experiment_df.to_csv(filename, index=False)
    experiment_df.to_json("experiment_results.json", orient="records")

    print(f"Experiment results saved to {filename}")

    # Network size
    policy_kwargs = dict(net_arch=[256, 256])

    # 0. Sanity check env
    print("\n[0] check_env...")
    tmp = CubeSatDetumblingEnv(render_mode=None, debug=False, plot_hist=False,
                              max_steps=MAX_STEPS, granularity=GRANULARITY, time_step=TIME_STEP)
    check_env(tmp, warn=True)
    tmp.close()
    print("[0] OK ✅\n")

    # 1. Optuna
    if RUN_OPTUNA:
        print("[1] Running Optuna...")
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study, elapsed = run_optuna(
            n_trials=N_TRIALS,
            device=DEVICE,
            study_name="dqn_cubesat",
            storage=None,  # if you want to persist: "sqlite:///optuna.db"
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

    # 2. Final training with auto-best-save
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

    # 3. Evaluate best model
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

    # 4. Save summary
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
        "final_training": {
            "timesteps": FINAL_TIMESTEPS,
            "eval_freq": EVAL_FREQ,
            "n_eval_episodes_during_train": N_EVAL_EPISODES,
            "policy_kwargs": policy_kwargs,
        },
         "best_params": best_params,
         "final_eval": eval_metrics,
         "paths": {
            "best_model": best_model_path,
            "last_model": last_model_path,
            "log_dir": LOG_DIR,
        },
    }

    # Load model
    model = DQN.load(best_model_path)
    print("Model loaded successfully.")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Gamma: {model.gamma}")
    print(f"Batch size: {model.batch_size}")
    print(f"Buffer size: {model.buffer_size}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=numpy_json_default)

    # Save training monitor plot
    monitor_path = "/home/mapacheroja/apr-lab-2/20260107_063620/logs/monitor.csv"

    # Plot training monitor
    plot_training_monitor(
        monitor_csv_path=monitor_path,
        window=50,
        save_dir=PLOTS_DIR,
        prefix="training"
    )

    # Plot final evaluation results
    plot_final_eval_from_json(
        json_path="20260107_063620/dqn_results.json",
        save_dir=PLOTS_DIR,
        prefix="final_eval"
    ) 
 
    best_model_path = "/home/mapacheroja/apr-lab-2/20260107_063620/best/best_model.zip"

    """
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
    """
