import os
import sys
import cv2
import time
import pickle
import random
import numpy as np
import gymnasium 
import vizdoom.gymnasium_wrapper
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv


class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shape, frame_skip):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = frame_skip
        print(env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        if observation.shape[-1] != 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        return observation


class GrayscaleObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
        )

    def observation(self, observation):
        if observation.shape[-1] != 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=-1)
        return gray


class DRL_Agent:
    def __init__(self, environment_id, learning_alg, train_mode=True, seed=None, n_envs=8, frame_skip=4):
        self.environment_id = environment_id
        self.learning_alg = learning_alg
        self.train_mode = train_mode
        self.seed = seed if seed else random.randint(0, 1000)
        self.policy_filename = f"{learning_alg}-{environment_id}-seed{self.seed}.policy.pkl"
        self.n_envs = n_envs if train_mode else 1
        self.frame_skip = frame_skip
        self.image_shape = (84, 84)
        self.training_timesteps = 10000
        self.num_test_episodes = 20
        self.l_rate = 0.00083
        self.gamma = 0.995
        self.n_steps = 512
        self.policy_rendering = True
        self.rendering_delay = 0.05 if self.environment_id.find("Vizdoom") > 0 else 0
        self.log_dir = './logs'
        self.model = None
        self.policy = None
        self.environment = None

        self._check_environment()
        self._create_log_directory()

        self.training_time = 0
        self.test_time = 0
        self.metrics = {}

    def _check_environment(self):
        available_envsA = [env for env in gymnasium.envs.registry.keys() if "LunarLander" in env]
        available_envsB = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
        if self.environment_id in available_envsA:
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsA}")
        elif self.environment_id in available_envsB:
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsB}")
        else:
            print(f"UNKNOWN environment={self.environment_id}")
            print(f"AVAILABLE_ENVS={available_envsA, available_envsB}")
            sys.exit(0)

    def _create_log_directory(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"Log directory created: {self.log_dir}")
        else:
            print(f"Log directory {self.log_dir} already exists!")

    def wrap_env(self, env):
        env = ObservationWrapper(env, shape=self.image_shape, frame_skip=self.frame_skip)
        env = GrayscaleObservationWrapper(env)
        if self.train_mode:
            env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    def create_environment(self, use_rendering=False):
        print("self.environment_id=" + str(self.environment_id))

        if self.environment_id.find("Vizdoom") == -1:
            if use_rendering:
                self.environment = gymnasium.make(self.environment_id, render_mode="human")
            else:
                self.environment = gymnasium.make(self.environment_id)
            self.environment = DummyVecEnv([lambda: self.environment])
            self.environment = VecMonitor(self.environment, self.log_dir)
            self.policy = "MlpPolicy"
        else:
            self.environment = make_vec_env(
                self.environment_id,
                n_envs=self.n_envs,
                seed=self.seed,
                monitor_dir=self.log_dir,
                wrapper_class=self.wrap_env
            )
            self.environment = VecFrameStack(self.environment, n_stack=4)
            self.environment = VecTransposeImage(self.environment)
            self.policy = "CnnPolicy"

        print("self.environment.action_space:", self.environment.action_space)

    def create_model(self):
        if self.learning_alg == "DQN":
            self.model = DQN(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma,
                             buffer_size=10000, batch_size=64, exploration_fraction=0.9, verbose=1)

        elif self.learning_alg == "A2C":
            self.model = A2C(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma,
                             verbose=1)

        elif self.learning_alg == "PPO":
            self.model = PPO(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma,
                             verbose=1)

        else:
            print(f"Unknown LEARNING_ALG={self.learning_alg}")
            sys.exit(0)

    def train_or_load_model(self):
        if self.train_mode:
            print(self.model)
            print("Training model...")
            start_time = time.time()
            self.model.learn(total_timesteps=self.training_timesteps)
            self.training_time = time.time() - start_time
            print(f"Training time: {self.training_time:.2f} seconds")
            print(f"Saving policy {self.policy_filename}")
            pickle.dump(self.model.policy, open(self.policy_filename, 'wb'))
        else:
            print("Loading policy...")
            with open(self.policy_filename, "rb") as f:
                policy = pickle.load(f)
            self.model.policy = policy

    def evaluate_policy(self):
        print("Evaluating policy...")
        start_time = time.time()
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=self.num_test_episodes)
        self.test_time = time.time() - start_time
        print(f"Test time: {self.test_time:.2f} seconds")
        self.metrics['Average Reward'] = mean_reward
        self.metrics['Reward StdDev'] = std_reward

    def render_policy(self):
        steps_per_episode = []
        reward_per_episode = []
        total_score = []

        episode = 1
        self.create_environment(True)
        env = self.environment
        obs = env.reset()

        print("DEMONSTRATION EPISODES:")
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if not hasattr(self, 'current_steps'):
                self.current_steps = 0
                self.current_reward = 0
                self.current_score = 0

            self.current_steps += 1
            self.current_reward += reward
            self.current_score += np.sum(reward)

            if any(done):
                print(f"episode={episode}, steps={self.current_steps}, reward={self.current_reward}")
                steps_per_episode.append(self.current_steps)
                reward_per_episode.append(self.current_reward)
                total_score.append(self.current_score)

                self.current_steps = 0
                self.current_reward = 0
                self.current_score = 0

                episode += 1
                obs = env.reset()

            if self.policy_rendering:
                env.render("human")
                time.sleep(self.rendering_delay)

            if episode > self.num_test_episodes:
                break
        env.close()

        self.metrics['Avg Steps per Episode'] = np.mean(steps_per_episode)
        self.metrics['Avg Game Score'] = np.mean(total_score)
        self.metrics['Training Time (s)'] = self.training_time
        self.metrics['Test Time (s)'] = self.test_time

        print("\nFINAL METRICS:")
        for key, value in self.metrics.items():
            print(f"{key}: {value:.2f}")

    def run(self):
        self.create_environment()
        self.create_model()
        self.train_or_load_model()
        self.evaluate_policy()
        self.render_policy()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("USAGE: sb-VizDoom.py (train|test) (DQN|A2C|PPO) [seed_number]")
        print("EXAMPLE1: sb-VizDoom.py train PPO")
        print("EXAMPLE2: sb-VizDoom.py test PPO 476")
        sys.exit(0)

    environment_id = "VizdoomTakeCover-v0"
    train_mode = sys.argv[1] == 'train'
    learning_alg = sys.argv[2]
    seed = random.randint(0, 1000) if train_mode else int(sys.argv[3])

    agent = DRL_Agent(environment_id, learning_alg, train_mode, seed)
    agent.run()
