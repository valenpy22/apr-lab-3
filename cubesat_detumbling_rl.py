# Libraries to use
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time


import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt
import matplotlib
from zmq.backend import second

plt.close('all')

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Import existing components from HoneySat simulator
from Simulations.RotationSimulation import RotationSimulation
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.MagneticSimulation import MagneticSimulation
from SatellitePersonality import SatellitePersonality


class CubeSatDetumblingEnv(gym.Env):
    """
    GYMNASIUM ENVIRONMENT FOR DETUMBLING PROBLEM USING HONEYSAT SIMULATOR.

    This environment integrates with existing classes of RotationSimulation, OrbitalSimulation
    and MagneticSimulation to provide a realistic simulation of satellite dynamics
    for reinforcement learning.
    """

    metadata = {'render_modes': ['human', 'none']}

    def __init__(self, render_mode=None, max_steps=10, start_time=datetime.now(), time_step=0.1, granularity=40, debug=False, num_bins=4, plot_hist=False, reward_scaling: float = 10.0):
        """
        Initialize the CubeSat environment for the detumbling problem.

        Args:
            render_mode (str): Render mode ('human' or None)
            max_steps (int): Maximum steps per episode
            start_time (datetime): Initial simulation time
            time_step (float): Simulation time step in seconds
            granularity (int): Granularity of the simulation, divides time_step
            debug (bool): Enable observation history and plotting
        """
        super().__init__()

        self._start_time = start_time
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.time_step = time_step
        self.current_time = start_time
        self.sim_granularity = granularity
        self._plot_hist = plot_hist
        self.reward_scaling = reward_scaling

        # initialize simulator components
        self.rotation_sim = None
        self.orbital_sim = None
        self.magnetic_sim = None

        self.num_bins = num_bins

        # Discretize the action space for Q-learning
        # Actions: Positive/Negative torque on each axis (X, Y, Z) + No torque
        self.max_torque = SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
        self.action_map = self.create_action_map_xyz()
        self.action_space = spaces.Discrete(len(self.action_map))
        #self.action_space = spaces.Discrete(2)
        #print(type(self.action_space))  # This should show <class 'gym.spaces.discrete.Discrete'>

        # define observation space
        # box: quaternion (4) + angular velocity (3) + magnetic field (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        # tracking within an episode
        self.current_step = 0
        self.episode_reward = 0.0

        # since it's almost impossible to have zero angular velocity, a threshold is set
        # adjustable depending on the mission and context
        self.success_threshold = 0.01  # rad/s

        # For debug effects, save observation history and plot them
        self._debug = debug  # Debug enabled
        self._observation_hist = []  # Observation history
        self._time_hist = []  # Historic time
        if self._debug:
            # import matplotlib.pyplot as plt
            pass
            # self.__figure, axes = plt.subplots(2, 1)
            # axes[0].grid(True)
            # axes[1].grid(True)
            # plt.ion()
            # plt.show(block=False)
    
    def create_action_map_xyz(self):
        """
        Discrete actions: torque 0 and +/- {T, T/2, T/4, T/8} on each axis (one at a time).
        Total actions: 1 + 3*(2*4) = 25
        """
        action_map = {}
        index = 0

        T = float(self.max_torque)

        # magnitude levels (fine control near 0)
        mags = np.array([T, T/2, T/4, T/8, T/16], dtype=np.float32)

        # 0 torque
        action_map[index] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        index += 1

        # X axis
        for m in mags:
            action_map[index] = np.array([+m, 0.0, 0.0], dtype=np.float32); index += 1
            action_map[index] = np.array([-m, 0.0, 0.0], dtype=np.float32); index += 1

        # Y axis
        for m in mags:
            action_map[index] = np.array([0.0, +m, 0.0], dtype=np.float32); index += 1
            action_map[index] = np.array([0.0, -m, 0.0], dtype=np.float32); index += 1

        # Z axis
        for m in mags:
            action_map[index] = np.array([0.0, 0.0, +m], dtype=np.float32); index += 1
            action_map[index] = np.array([0.0, 0.0, -m], dtype=np.float32); index += 1
            
        # print(action_map)

        return action_map



    def _create_simulators(self):
        """Create simulator instances.
        - RotationSimulation
        - OrbitalSimulation
        - MagneticSimulation
        """
        if self.rotation_sim is not None:
            try:
                self.rotation_sim.stop()
            except Exception:
                pass
        if self.orbital_sim is not None:
            try:
                self.orbital_sim.stop()
            except Exception:
                pass
        if self.magnetic_sim is not None:
            try:
                self.magnetic_sim.stop()
            except Exception:
                pass

        # call simulator constructors, check parameters if debugging is needed
        self.rotation_sim = RotationSimulation(debug=False)
        self.orbital_sim = OrbitalSimulation(self.rotation_sim)
        self.magnetic_sim = MagneticSimulation(self.orbital_sim, self.rotation_sim)

    def _start_simulators(self):
        """Initialize threads of each simulator. Parallelized implementation."""
        #TODO: Do not start the simulation thread, control them manually instead
        try:
            self.rotation_sim.start()
            self.orbital_sim.start()
            self.magnetic_sim.start()
        except Exception as e:
            print(f"Warning: Could not start all simulators: {e}")

    def _stop_simulators(self):
        """Stop the threads of each simulator."""
        try:
            if self.rotation_sim:
                self.rotation_sim.stop()
            if self.orbital_sim:
                self.orbital_sim.stop()
            if self.magnetic_sim:
                self.magnetic_sim.stop()
        except Exception as e:
            print(f"Warning: Error stopping simulators: {e}")

    def reset(self, seed=None, options=None):
        """
        Function to reset the environment and start a new episode.
        Args:
            seed (int): Random seed for reproducibility
            options (dict): Additional options (not used)

        Returns:
            tuple: (observation, information)
        """
        super().reset(seed=seed)

        start_time = getattr(self, '_start_time', None)

        if self.start_time is None:
            self.current_time = datetime.datetime(2025, 1, 1)
        else:
            self.current_time = self.start_time

        # stop simulations to then restart them for new episode
        self._stop_simulators()
        self._create_simulators()

        # random or fixed initial conditions
        initial_angular_velocity = self.np_random.uniform(-1.0, 1.0, size=3)
        # initial_angular_velocity = self.np_random.uniform(-1.0, 1.0, size=3)
        # initial_angular_velocity = np.array([1.0, 0.0, 0.0])

        initial_quat = self.np_random.normal(size=4)
        # initial_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Set initial conditions
        initial_quat /= np.linalg.norm(initial_quat)
        self.rotation_sim.angular_velocity = initial_angular_velocity
        self.rotation_sim.quaternion = initial_quat

        # start simulations with new conditions
        self._start_simulators()

        # reset tracking
        self.current_step = 0
        self.episode_reward = 0.0

        # wait for everything to restart, not the best solution but it works
        time.sleep(0.1)

        observation = self._get_observation()
        info = {}

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        """
        Executes a single step in the environment within an episode.

        Args:
            action (np.ndarray): 3-dimensional command representing torque

        Returns:
            tuple: (observation, reward, terminated, truncated, information)
        """
        # map discrete action to torque vector
        torque_action = self.action_map[action]
        # print("torque action", torque_action)

        ### TEST: Compare with a simple proportional controller
        # G = 1e-3
        # torque_action = -self.rotation_sim.angular_velocity*G
        ###

        # 🔹 SAVE PREVIOUS STATE BEFORE APPLYING ACTION
        previous_angular_vel_norm = np.linalg.norm(self.rotation_sim.angular_velocity)

        # apply torque action to rotation simulator
        self.rotation_sim.set_torque(torque_action)

        # Advance the simulation with smaller granularity
        dt = self.time_step / self.sim_granularity
        for i in range(self.sim_granularity):
            self.current_time += timedelta(seconds=dt) # Advance time in the defined step
            self.rotation_sim.update_simulation(dt) # Update the simulation

            # get new observation
            observation = self._get_observation()
            # Save history for plotting
            if self._debug:
                # Add torque also to history
                observation = np.concatenate((observation, torque_action))
                self._observation_hist.append(observation)
                # Add time to history
                self._time_hist.append(self.current_time.timestamp())

        # 🔹 CALCULATE REWARD WITH PREVIOUS STATE
        reward = self._calculate_reward(torque_action, previous_angular_vel_norm)
        self.episode_reward += reward

        # check if the episode has finished
        try:
            angular_vel_norm = np.linalg.norm(self.rotation_sim.angular_velocity)
            terminated = angular_vel_norm < self.success_threshold
            terminated = bool(terminated)
        except Exception:
            angular_vel_norm = 1.0
            terminated = False

        # check if an episode timeout occurs
        # "truncated" parameter (check gymnasium docs)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        truncated = bool(truncated)

        # return additional info
        info = {
            'angular_velocity_norm': angular_vel_norm,
            'episode_reward': self.episode_reward,
            'success': bool(terminated),
        }

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Get the current observation from the simulator.

        Returns:
            np.ndarray: Observation vector: [quat(4), angular_vel(3), mag_field(3)]
        """
        try:
            # get state from rotation simulator
            quaternion = self.rotation_sim.quaternion.copy()
            angular_velocity = self.rotation_sim.angular_velocity.copy()

            # get magnetic field info
            try:
                mag_field_data = self.magnetic_sim.send_request('earth_magnetic_field').result()
                # extract x, y, z components and convert from nT to T
                mag_field_inertial = np.array([
                    mag_field_data['north'],
                    mag_field_data['east'],
                    mag_field_data['vertical']
                ]) * 1e-9

                # rotate magnetic field from inertial to body using quaternion
                mag_field_body = self._rotate_vector_by_quaternion(mag_field_inertial, quaternion)

            except Exception as e:
                print(f"Warning: Could not get magnetic field: {e}")
                mag_field_body = np.zeros(3)

            observation = np.concatenate([
                quaternion,
                angular_velocity,
                mag_field_body
            ]).astype(np.float32)

            observation = observation.flatten()

            #print("Observation shape:", observation.shape)
            #print("Observation:", observation)

            return observation

        except Exception as e:
            print(f"Error getting observation: {e}")
            # return default observation in case of failure
            return np.zeros(10, dtype=np.float32)

    def _rotate_vector_by_quaternion(self, vector, quaternion):
        """
        Rotate a vector from inertial frame to body frame using a quaternion.
        Wrapper of what already exists in RotationSimulation.

        Args:
            vector (np.ndarray): 3d vector in inertial frame
            quaternion (np.ndarray): Quaternion [qx, qy, qz, qw]

        Returns:
            np.ndarray: Rotated vector in body frame
        """
        try:
            # represent vector as a pure quaternion
            v_quat = np.array([vector[0], vector[1], vector[2], 0.0])

            # quaternion conjugate (inertial to body)
            q_conj = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])

            # Use existing static method for quaternion multiplication
            # rotated_v = q_conj * v_quat * q
            temp = RotationSimulation.quat_mut(q_conj, v_quat)
            rotated_v = RotationSimulation.quat_mut(temp, quaternion)

            # return only the vector part
            return rotated_v[:3]

        except Exception as e:
            print(f"Warning: Quaternion rotation failed: {e}")
            return vector

    def _calculate_reward(self, action, previous_angular_vel_norm, reward_scaling=1.0):
        """
        Reward: combines
        - objective: minimize ||ω||
        - shaping: reward improvement (decrease of ||ω||)
        - control: penalize torque, stronger when already close
        - precision: small continuous bonus when in fine regime
        - success: large bonus when crossing threshold
        """

        try:
            angular_vel_norm = float(np.linalg.norm(self.rotation_sim.angular_velocity))
        except Exception:
            angular_vel_norm = 1.0

        control_effort = float(np.linalg.norm(action))

        # 1) Base: penalizes current magnitude
        base_reward = -angular_vel_norm

        # 2) Shaping for improvement (if ||ω|| decreases, positive)
        improvement = previous_angular_vel_norm - angular_vel_norm
        shaped_reward = self.reward_scaling * improvement

        # 3) Adaptive control penalty:
        #    far: soft; near: strong (to avoid over-control and oscillation)
        if angular_vel_norm < 0.2:
            control_penalty = -0.05 * control_effort
        else:
            control_penalty = -0.01 * control_effort

        # 4) Success bonus (crossing threshold)
        success_bonus = 50.0 if angular_vel_norm < self.success_threshold else 0.0

        # 5) Total reward: scaling applied here
        reward = base_reward + shaped_reward + control_penalty + success_bonus

        return float(reward)



    def render(self):
        """
        Render current state of the environment.
        """
        if self.render_mode == 'human':
            try:
                quaternion = self.rotation_sim.quaternion
                angular_velocity = self.rotation_sim.angular_velocity
                angular_vel_norm = np.linalg.norm(angular_velocity)

                # print(f"Step: {self.current_step:3d} | "
                #       f"Time: {self.current_time.timestamp():.4f} | "
                #       f"ω_norm: {angular_vel_norm:.6f} rad/s | "
                #       f"ω: [{angular_velocity[0]:.6f}, {angular_velocity[1]:.6f}, {angular_velocity[2]:.6f}] rad/s | "
                #       f"Episode Reward: {self.episode_reward:.2f} | "
                #       f"Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            except Exception as e:
                print(f"Render error: {e}")

        if self.render_mode == 'plot':
            pass

    def close(self):
        """
        Clean environment and restart all external simulators.
        """
        if self._plot_hist:
            self.show_hist()
        self._stop_simulators()

    def show_hist(self):
        if len(self._observation_hist) == 0:
            print("No saved history")
            return

        observation_hist = np.array(self._observation_hist)
        quat_hist = observation_hist[:,0:4]
        vel_hist = observation_hist[:,4:7]
        mag_hist = observation_hist[:,7:10]
        torque_hist = observation_hist[:,10:13]

        figure, axes = plt.subplots(3, 1)
        plt.title("Rotation Simulation")
        axes[0].grid(True)
        axes[1].grid(True)
        plt.ion()
        plt.show(block=False)

        axes[0].clear()
        axes[0].plot(self._time_hist, np.array(vel_hist), "--.", label=["x", "y", "z"])
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel('Velocity (rad/s)')
        axes[0].set_xlabel('Time')
        axes[0].grid(True)

        axes[1].clear()
        axes[1].plot(self._time_hist, np.array(quat_hist), "--.", label=["i", "j", "k", "s"])
        axes[1].legend(loc="upper right")
        axes[1].set_ylabel('Quaternion')
        axes[1].set_xlabel('Time')
        axes[1].grid(True)

        axes[2].clear()
        axes[2].plot(self._time_hist, np.array(torque_hist), "--.", label=["Tx", "Ty", "Tz"])
        axes[2].legend(loc="upper right")
        axes[2].set_ylabel('Torque (%)')
        axes[2].set_xlabel('Time')
        axes[2].grid(True)

        plt.show(block=True)


def test_environment_basic():
    """
    Test function to demonstrate environment operation.
    """
    print("=" * 60)
    print("Testing CubeSat Detumbling Environment")
    print("=" * 60)

    env = CubeSatDetumblingEnv(render_mode='human')

    try:
        # reset environment
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")

        # take 10 random actions
        for i in range(10):
            action = env.action_space.sample()
            print(f"\nStep {i + 1}: Action = {action}")

            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"Reward: {reward:.4f}")

            if terminated or truncated:
                print(f"Episode ended at step {i + 1}")
                if terminated:
                    print("SUCCESS: Detumbling achieved!")
                else:
                    print("Episode truncated (timeout)")
                break

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment test completed!")

def evaluate_random_agent(episodes=100, max_steps=400):
    """
    Evaluates the performance of a random agent (baseline) and records the history
    of rewards and successes per episode.

    Args:
        episodes (int): Number of episodes for evaluation.
        max_steps (int): Maximum steps per episode.

    Returns:
        tuple: (final_metrics, reward_history, success_history)
    """
    print("=" * 70)
    print(f"🧪 RANDOM AGENT EVALUATION (BASELINE) - {episodes} EPISODES")
    print("=" * 70)

    # Create an environment instance without rendering for evaluation
    env = CubeSatDetumblingEnv(render_mode=None, max_steps=max_steps, debug=False, plot_hist=False)
    
    total_rewards = []
    success_count = 0
    steps_to_success = []
    
    # Lists to store the history of each episode
    episode_rewards_hist = []
    episode_success_hist = [] 
    
    start_time = time.time()

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 1. Action Selection: Random (uniform probability)
            action = env.action_space.sample()
            
            # 2. Step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        # 3. Metrics and history recording
        total_rewards.append(total_reward)
        episode_rewards_hist.append(total_reward)
        
        is_success = terminated
        episode_success_hist.append(is_success)
        
        if terminated:
            success_count += 1
            steps_to_success.append(step_count)
            
        if (episode + 1) % 10 == 0 or episode == episodes - 1:
            print(f"  Episode {episode + 1}/{episodes} | Reward: {total_reward:.2f} | Success: {'YES' if terminated else 'NO'}")

    end_time = time.time()
    
    # 4. Final Metrics Calculation
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = success_count / episodes
    
    if success_count > 0:
        avg_steps_success = np.mean(steps_to_success)
        std_steps_success = np.std(steps_to_success)
    else:
        avg_steps_success = np.nan
        std_steps_success = np.nan

    env.close()

    metrics = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'avg_steps_on_success': avg_steps_success,
        'std_steps_on_success': std_steps_success,
        'total_time_s': end_time - start_time
    }

    print("\n" + "=" * 70)
    print("📊 METRICS SUMMARY (RANDOM BASELINE)")
    print("-" * 70)
    print(f"Average Reward (μ): **{metrics['mean_reward']:.2f}**")
    print(f"Std. Deviation (σ):       {metrics['std_reward']:.2f}")
    print(f"Success Rate:            **{metrics['success_rate'] * 100:.2f}%**")
    print(f"Avg. Steps to Success:     {metrics['avg_steps_on_success']:.1f} (only if there are successes)")
    print("-" * 70)
    print(f"Total simulation time: {metrics['total_time_s']:.2f} seconds")
    print("=" * 70)

    # Return final metrics and histories
    return metrics, episode_rewards_hist, episode_success_hist

def plot_random_agent_history(episode_rewards, episode_success, window_size=10):
    """
    Generates plots of the random agent execution.

    Args:
        episode_rewards (list): List of accumulated rewards per episode.
        episode_success (list): List of booleans indicating success per episode.
        window_size (int): Window size for moving average.
    """
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # 1. Smooth the Reward (Moving Average)
    rewards_series = np.array(episode_rewards)
    # Simple moving average
    smoothed_rewards = np.convolve(rewards_series, np.ones(window_size)/window_size, mode='valid')
    # Adjust X axis for moving average
    episodes_smoothed = np.arange(window_size, len(episode_rewards) + 1)

    # 2. Calculate Cumulative Success Rate
    success_rate_cumulative = np.cumsum(episode_success) / episodes

    # 3. Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    fig.suptitle('Random Agent Metrics (Baseline)', fontsize=16, fontweight='bold')

    # --- Plot 1: Reward per Episode (Smoothed) ---
    axes[0].plot(episodes_smoothed, smoothed_rewards, label=f'Moving Average ({window_size} eps)', color='orange')
    axes[0].set_ylabel('Accumulated Reward (Moving Average)')
    axes[0].set_title('Reward Progress per Episode')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # --- Plot 2: Cumulative Success Rate ---
    axes[1].plot(episodes, success_rate_cumulative, label='Cumulative Success Rate', color='green')
    axes[1].set_ylabel('Cumulative Success Rate')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    
    # Final success rate line
    if success_rate_cumulative.size > 0:
        final_rate = success_rate_cumulative[-1]
        axes[1].axhline(y=final_rate, color='r', linestyle=':', label=f'Final Rate: {final_rate:.2f}')
    
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

    print("\nBaseline plots generated.")

if __name__ == "__main__":
    """
    Simple test in this same script.
    For training, it is recommended to use the script train_cubesat_detumbling.py.
    """
    print("=" * 60)
    print("CubeSat Detumbling Environment Test")
    print("=" * 60)

    debug = True  # Enable or disable plots
    plot_hist = True
    start_time = datetime.fromtimestamp(1758566834)
    time_step = 1
    total_time = 15*60
    granularity = 10

    # create and test environment
    env = CubeSatDetumblingEnv(render_mode='human', start_time=start_time, time_step=time_step, granularity=granularity,
                               debug=debug, plot_hist=plot_hist)

    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    """
    try:
        # run a test episode...
        obs, _ = env.reset()
        print(f"\nInitial observation shape: {obs.shape}")
        print("Running N random steps...")

        for step in np.arange(0, total_time, time_step):
            action = env.action_space.sample()
            print(f"Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward:.4f}")

            if terminated:
                print(f"\nSUCCESS! Episode completed at step {step + 1}")
                break
            elif truncated:
                print(f"\nEpisode truncated at step {step + 1}")
                break

    except Exception as e:
        print(f"Test error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment test completed!")
    

    """
    # --- Configuración para la Línea Base ---
    EPISODES_TO_EVALUATE = 10 
    MAX_STEPS_PER_EPISODE = 400 # Asegúrate de que coincida con la configuración por defecto
    
    print("=" * 60)
    print("CubeSat Detumbling Environment Test")
    print("=" * 60)
    
    try:
        # 1. Execute Random Agent and get metrics AND HISTORY
        # We capture the 3 returned values: final metrics, reward history and success history.
        random_agent_metrics, rewards_hist, success_hist = evaluate_random_agent(
            episodes=EPISODES_TO_EVALUATE, 
            max_steps=MAX_STEPS_PER_EPISODE
        )
        
        # 2. Generate history plots
        # We use the histories captured in step 1.
        plot_random_agent_history(rewards_hist, success_hist, window_size=10)
        
        # 3. Print the results
        print("\n🎉 Baseline Evaluation completed.")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

    # Close the initial environment instance if not used in try/except block.
    # If you used the commented test block above, this would close its instance.
    try:
        env.close()
    except Exception:
        pass
    
    # Note: It is recommended to keep this test section separate from the real 
    # training code for a cleaner structure.
    