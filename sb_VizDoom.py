
# STUDENT NAME: CHIZOBA GERALDINE OKABUONYE
# STUDENT ID : 29653331
# COURSE TITLE: CMP9137 ADVANCED MACHINE LEARNING

# Importing necessary Python libraries
import os
import sys
import cv2  # For image processing
import time
import pickle  # For saving and loading the model
import random
import numpy as np  # For numerical operations
import gymnasium  # Reinforcement learning environment library
import vizdoom.gymnasium_wrapper  # Wrapper for VizDoom environments

# Importing stable-baselines3 for RL algorithms
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom wrapper to resize and format observations (images) from the environment
class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shape, frame_skip):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]  # Reversing shape for OpenCV
        self.env.frame_skip = frame_skip  # How many frames to skip
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)  # Resize the screen image
        if observation.shape[-1] != 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)  # Ensure RGB format
        return observation

# Converts RGB image to grayscale
class GrayscaleObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(0, 255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)

    def observation(self, observation):
        if observation.shape[-1] != 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        gray = np.expand_dims(gray, axis=-1)  # Add channel dimension
        return gray

# Main class to manage the reinforcement learning agent
class DRL_Agent:
    def __init__(self, environment_id, learning_alg, train_mode=True, seed=None, n_envs=8, frame_skip=4):
        # Set basic parameters and hyperparameters
        self.environment_id = environment_id
        self.learning_alg = learning_alg
        self.train_mode = train_mode
        self.seed = seed if seed else random.randint(0, 1000)  # Random seed if not provided
        self.policy_filename = f"{learning_alg}-{environment_id}-seed{self.seed}.policy.pkl"
        self.n_envs = n_envs if train_mode else 1
        self.frame_skip = frame_skip
        self.image_shape = (84, 84)  # Resize all observations to this shape
        self.training_timesteps = 500_000
        self.num_test_episodes = 50
        self.l_rate = 0.0003
        self.gamma = 0.99  # Discount factor
        self.n_steps = 2048
        self.policy_rendering = True
        self.rendering_delay = 0.05 if self.environment_id.find("Vizdoom") > 0 else 0
        self.log_dir = './logs'  # Folder to save logs
        self.model = None
        self.policy = None
        self.environment = None

        # Reduce test load if not training
        if not self.train_mode:
            self.num_test_episodes = 10
            self.policy_rendering = False
            self.rendering_delay = 0.05

        self._check_environment()  # Make sure environment is valid
        self._create_log_directory()  # Make sure log folder exists

        self.training_time = 0
        self.test_time = 0
        self.metrics = {}  # Store metrics like rewards, time, etc.

    # Check if the provided environment exists
    def _check_environment(self):
        available_envsA = [env for env in gymnasium.envs.registry.keys() if "LunarLander" in env]
        available_envsB = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
        if self.environment_id in available_envsA or self.environment_id in available_envsB:
            print(f"Environment {self.environment_id} found.")
        else:
            print(f"Unknown environment={self.environment_id}")
            sys.exit(0)

    # Create directory for saving logs
    def _create_log_directory(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    # Wraps environment with preprocessing
    def wrap_env(self, env):
        env = ObservationWrapper(env, shape=self.image_shape, frame_skip=self.frame_skip)
        env = GrayscaleObservationWrapper(env)
        if self.train_mode:
            env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)  # Scale rewards
        return env

    # Create and prepare environment
    def create_environment(self, use_rendering=False):
        if self.environment_id.find("Vizdoom") == -1:
            # For standard environments
            if use_rendering:
                self.environment = gymnasium.make(self.environment_id, render_mode="human")
            else:
                self.environment = gymnasium.make(self.environment_id)
            self.environment = DummyVecEnv([lambda: self.environment])
            self.environment = VecMonitor(self.environment, self.log_dir)
            self.policy = "MlpPolicy"
        else:
            # For VizDoom environments
            self.environment = make_vec_env(
                self.environment_id,
                n_envs=self.n_envs,
                seed=self.seed,
                monitor_dir=self.log_dir,
                wrapper_class=self.wrap_env
            )
            self.environment = VecFrameStack(self.environment, n_stack=4)  # Stack 4 frames for better temporal context
            self.environment = VecTransposeImage(self.environment)  # Change shape to CHW format
            self.policy = "CnnPolicy"

    # Create model based on the selected learning algorithm
    def create_model(self):
        if self.learning_alg == "DQN":
            self.model = DQN(
                self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma,
                buffer_size=50000, batch_size=128, exploration_fraction=0.3,
                exploration_initial_eps=1.0, exploration_final_eps=0.1,
                verbose=1
            )
        elif self.learning_alg == "A2C":
            self.model = A2C(
                self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate,
                gamma=self.gamma, n_steps=self.n_steps, verbose=1
            )
        elif self.learning_alg == "PPO":
            self.model = PPO(
                self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate,
                gamma=self.gamma, n_steps=self.n_steps, verbose=1
            )
        else:
            print(f"Unknown learning algorithm={self.learning_alg}")
            sys.exit(0)

    # Either trains the model or loads a pre-trained one
    def train_or_load_model(self):
        if self.train_mode:
            start_time = time.time()
            self.model.learn(total_timesteps=self.training_timesteps)  # Start training
            self.training_time = time.time() - start_time
            with open(self.policy_filename, "wb") as f:
                pickle.dump(self.model.policy, f)  # Save policy
        else:
            with open(self.policy_filename, "rb") as f:
                policy = pickle.load(f)  # Load saved policy
            self.model.policy = policy

    # Evaluate how well the model performs
    def evaluate_policy(self):
        mean_reward, std_reward = evaluate_policy(
            self.model, self.model.get_env(), n_eval_episodes=self.num_test_episodes
        )
        self.metrics['Average Reward'] = mean_reward
        self.metrics['Reward StdDev'] = std_reward

    # Run the model and show its performance
    def render_policy(self):
        steps_per_episode = []
        reward_per_episode = []
        total_score = []

        episode = 1
        self.create_environment(True)  # Enable rendering
        env = self.environment
        obs = env.reset()

        start_time = time.time()

        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Initialize episode-specific counters (steps, reward, score) if not already done

            if not hasattr(self, 'current_steps'):
                self.current_steps = 0
                self.current_reward = 0
                self.current_score = 0

            self.current_steps += 1
            self.current_reward += reward
            self.current_score += np.sum(reward)

            # Episode finished
            if any(done):
                steps_per_episode.append(self.current_steps)
                reward_per_episode.append(self.current_reward)
                total_score.append(self.current_score)

                self.current_steps = 0
                self.current_reward = 0
                self.current_score = 0

                episode += 1
                obs = env.reset()

            if self.policy_rendering:
                env.render("human")  # Show game window
                time.sleep(self.rendering_delay)

            if episode > self.num_test_episodes:
                break

        self.test_time = time.time() - start_time
        env.close()

        # Log final metrics
        self.metrics['Avg Steps per Episode'] = np.mean(steps_per_episode)
        self.metrics['Avg Game Score'] = np.mean(total_score)
        self.metrics['Training Time (s)'] = self.training_time
        self.metrics['Test Time (s)'] = self.test_time

        print("\nFINAL METRICS:")
        for key, value in self.metrics.items():
            print(f"{key}: {value:.4f}")

    # Full run pipeline: environment, model, train/load, evaluate, render
    def run(self):
        self.create_environment()
        self.create_model()
        self.train_or_load_model()
        self.evaluate_policy()
        self.render_policy()

# Entry point of the script
if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("USAGE: sb-VizDoom.py (train|test) (DQN|A2C|PPO) [seed_number]")
        sys.exit(0)

    environment_id = "VizdoomTakeCover-v0"
    train_mode = sys.argv[1] == 'train'
    learning_alg = sys.argv[2]
    seed = random.randint(0, 1000) if train_mode else int(sys.argv[3])

    # Create and run the agent
    agent = DRL_Agent(environment_id, learning_alg, train_mode, seed)
    agent.run()
