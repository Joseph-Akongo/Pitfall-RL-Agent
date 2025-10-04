"""
Pitfall DQN Trainer
-------------------
A reinforcement learning (RL) agent using Double DQN to learn
to play the Atari 2600 game Pitfall! via the Gymnasium toolkit.
"""

import os
import random
import datetime
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import gymnasium as gym
from gymnasium.wrappers import FrameStack, AtariPreprocessing

# ============================================================
# CONFIGURATION
# ============================================================
STRICT_MODE = True  # True = authentic Pitfall scoring and 20-min cap
FRAME_SKIP = 4
MAX_STEPS = 10_000_000  # 20 minutes at 60 FPS

# Reward shaping settings
USE_LIFE_PENALTY = True
USE_TIME_PENALTY = not STRICT_MODE
USE_ROOM_BONUS = True
USE_VINE_BONUS = True

LIFE_PENALTY = -1.0
TIME_PENALTY = -0.001
ROOM_BONUS = 0.2
VINE_BONUS = 0.05
NUM_EPISODES = 50_000

USE_AMP = True  # Use CUDA mixed precision if available

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def convert_obs_to_chw(obs) -> np.ndarray:
    """Convert Gym observation to (C, H, W) uint8 format."""
    arr = np.asarray(obs)
    if arr.ndim == 4:  # Sometimes comes batched
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D observation, got {arr.shape}")

    # Convert HWC → CHW if needed
    if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.uint8)


def set_global_seed(seed: int):
    """Make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ============================================================
# CUSTOM GYM WRAPPERS
# ============================================================
class AutoFireOnReset(gym.Wrapper):
    """Automatically press FIRE after resetting (some Atari games require this)."""

    def __init__(self, env, fire_frames: int = 10):
        super().__init__(env)
        self.fire_frames = fire_frames
        self.fire_action = None

    def _find_fire_action(self):
        if self.fire_action is not None:
            return
        if hasattr(self.env.unwrapped, "get_action_meanings"):
            actions = self.env.unwrapped.get_action_meanings()
            self.fire_action = actions.index("FIRE") if "FIRE" in actions else 0
        else:
            self.fire_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._find_fire_action()
        for _ in range(self.fire_frames):
            obs, _, term, trunc, _ = self.env.step(self.fire_action)
            if term or trunc:
                obs, info = self.env.reset(**kwargs)
        return obs, info


# ============================================================
# DQN NETWORK
# ============================================================
class QNetwork(nn.Module):
    """Convolutional network for estimating Q-values (Double DQN)."""

    def __init__(self, channels: int, height: int, width: int, n_actions: int):
        super().__init__()
        if height != 84 or width != 84:
            raise ValueError(f"Expected input 84x84, got {height}x{width}")

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            flat_features = self.cnn(torch.zeros(1, channels, height, width)).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.fc(self.cnn(x))


# ============================================================
# REPLAY BUFFER
# ============================================================
class ReplayBuffer:
    """Stores and samples past experience for training."""

    def __init__(self, capacity: int = 1_000_000):
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(np.float32(reward))
        self.next_states.append(next_state.copy())
        self.dones.append(np.uint8(done))

    def sample(self, batch_size: int, device: torch.device):
        """Sample a random batch of experiences."""
        idxs = random.sample(range(len(self.states)), batch_size)

        s = np.stack([self.states[i] for i in idxs])
        sn = np.stack([self.next_states[i] for i in idxs])
        a = np.array([self.actions[i] for i in idxs], dtype=np.int64)
        r = np.array([self.rewards[i] for i in idxs], dtype=np.float32)
        d = np.array([self.dones[i] for i in idxs], dtype=np.float32)

        s = torch.from_numpy(s).float().div_(255.0).to(device)
        sn = torch.from_numpy(sn).float().div_(255.0).to(device)
        a = torch.from_numpy(a).to(device)
        r = torch.from_numpy(r).to(device)
        d = torch.from_numpy(d).to(device)

        return s, a, r, sn, d

    def __len__(self):
        return len(self.states)


# ============================================================
# AGENT (Double DQN)
# ============================================================
@dataclass
class AgentSettings:
    gamma: float = 0.99
    sync_target_every: int = 50_000
    learn_every: int = 1_000
    batch_size: int = 64
    warmup_steps: int = 30_000
    learning_rate: float = 2.5e-4
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 10_000_000


class Agent:
    """DQN Agent that interacts with the environment and learns from experience."""

    def __init__(self, state_shape, n_actions: int, save_path: Path, cfg: AgentSettings = AgentSettings()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = state_shape[-3:]

        self.actions_count = n_actions
        self.config = cfg
        self.step_count = 0
        self.memory = ReplayBuffer(2_000_000)
        self.save_path = save_path
        self.save_interval = 10_000

        self.policy_net = QNetwork(c, h, w, n_actions).to(self.device)
        self.target_net = QNetwork(c, h, w, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.scaler = torch.amp.GradScaler(enabled=USE_AMP and self.device.type == "cuda")

    # -------------------------------------------------------
    # Epsilon-greedy exploration
    # -------------------------------------------------------
    def current_epsilon(self):
        progress = min(1.0, self.step_count / self.config.eps_decay_steps)
        return (1.0 - progress) * self.config.eps_start + progress * self.config.eps_end

    # -------------------------------------------------------
    # Choose an action
    # -------------------------------------------------------
    @torch.inference_mode()
    def choose_action(self, state):
        if random.random() < self.current_epsilon():
            self.step_count += 1
            return random.randrange(self.actions_count)
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().div_(255.0).to(self.device)
        q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        self.step_count += 1
        return action

    # -------------------------------------------------------
    # Learn from replay memory
    # -------------------------------------------------------
    def learn(self):
        if len(self.memory) < max(self.config.warmup_steps, self.config.batch_size):
            return None, None
        if self.step_count % self.config.learn_every != 0:
            return None, None

        s, a, r, sn, d = self.memory.sample(self.config.batch_size, self.device)

        with torch.amp.autocast("cuda", enabled=USE_AMP and self.device.type == "cuda"):
            q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            next_actions = torch.argmax(self.policy_net(sn), dim=1)
            next_q_values = self.target_net(sn).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = r + (1.0 - d) * self.config.gamma * next_q_values
            loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.step_count % self.config.sync_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.step_count % self.save_interval == 0:
            self.save_checkpoint()

        return float(loss.item())

    # -------------------------------------------------------
    # Save model
    # -------------------------------------------------------
    def save_checkpoint(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        file = self.save_path / f"dqn_checkpoint_{int(self.step_count)}.pth"
        torch.save(self.policy_net.state_dict(), file)
        print(f"💾 Saved model to {file}")


# ============================================================
# ENVIRONMENT CREATION
# ============================================================
def create_env(render=False):
    """Create a preprocessed Pitfall! environment."""
    env = gym.make("ALE/Pitfall-v5", render_mode=("human" if render else "rgb_array"), frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=FRAME_SKIP, scale_obs=False)
    env = FrameStack(env, num_stack=4)
    env = AutoFireOnReset(env)
    return env


# =========================
# Logger
# =========================
class MetricLogger:
    def __init__(self, save_dir: Path):
        self.log_file = save_dir / "log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"{'Episode':>8}{'Step':>12}{'Epsilon':>12}{'MeanR@100':>12}"
                    f"{'MeanLen@100':>14}{'MeanLoss@100':>14}{'MeanQ@100':>12}{'Time':>22}\n")
        self.ep_rewards, self.ep_lengths, self.ep_losses = [], [], []
        self._reset_episode()

    def _reset_episode(self):
        self.curr_reward, self.curr_length, self.curr_loss_sum, self.curr_loss_count = 0, 0, 0.0, 0

    def log_step(self, reward, loss, _q):
        self.curr_reward += reward
        self.curr_length += 1
        if loss is not None:
            self.curr_loss_sum += loss
            self.curr_loss_count += 1

    def log_episode(self):
        avg_loss = self.curr_loss_sum / self.curr_loss_count if self.curr_loss_count else 0.0
        self.ep_rewards.append(self.curr_reward)
        self.ep_lengths.append(self.curr_length)
        self.ep_losses.append(avg_loss)
        self._reset_episode()

    def record(self, episode, epsilon, step):
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        mean_r = np.mean(self.ep_rewards[-100:])
        mean_len = np.mean(self.ep_lengths[-100:])
        mean_loss = np.mean(self.ep_losses[-100:])
        with open(self.log_file, "a") as f:
            f.write(f"{episode:8d}{step:12d}{epsilon:12.3f}{mean_r:12.3f}"
                    f"{mean_len:14.1f}{mean_loss:14.4f}{0.0:12.3f}{now:>22}\n")
        print(f"Ep {episode:4d} | step {step:8d} | eps {epsilon:5.3f} | "
              f"R@100 {mean_r:7.2f} | L@100 {mean_len:6.1f} | loss@100 {mean_loss:7.4f}")

# ============================================================
# TRAINING LOOP
# ============================================================
def train_agent():
    """Main training loop for the Pitfall agent."""
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # Set up folder for saving results
    save_path = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_path.mkdir(parents=True, exist_ok=True)

    #create logger
    logger = MetricLogger(save_path)

    env = create_env(render=True)
    obs, _ = env.reset(seed=42)
    state = convert_obs_to_chw(obs)
    n_actions = env.action_space.n

    agent = Agent(state.shape, n_actions, save_path)
    total_steps = 0

    for episode in range(1, NUM_EPISODES + 1):  # short test run; adjust for full training
        obs, _ = env.reset()
        state = convert_obs_to_chw(obs)
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = convert_obs_to_chw(next_obs)
            agent.memory.store(state, action, reward, next_state, done)

            loss = agent.learn()
            result = agent.learn()
            if result is not None:
                # unpack if it's a tuple
                if isinstance(result, tuple):
                    _, loss = result
                else:
                    loss = result
            else:
                loss = None
            logger.log_step(reward, loss, 0.0)

            episode_reward += reward
            state = next_state
            total_steps += 1

            if total_steps > MAX_STEPS:
                done = True

        logger.log_episode()
        if episode % 10 == 0:
            logger.record(episode, agent.current_epsilon(), agent.step_count)

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {total_steps}")

    env.close()
    print("Training finished")

if __name__ == "__main__":
    train_agent()
