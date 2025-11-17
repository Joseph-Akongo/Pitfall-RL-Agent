"""
train.py
--------
Train a Double DQN agent to play Atari Pitfall!
Focus: Survival, exploration, and learning timing patterns.

Usage:
  python train.py                    # Start new training
  python train.py --resume PATH      # Continue from checkpoint
"""

import argparse
import datetime
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStack, AtariPreprocessing
from torch import nn

from config import *


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def set_global_seed(seed: int):
    """Make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def convert_obs_to_chw(obs) -> np.ndarray:
    """Convert Gym observation to (C, H, W) uint8 format."""
    arr = np.asarray(obs)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D observation, got {arr.shape}")
    if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.uint8)


# ============================================================
# CUSTOM GYM WRAPPERS
# ============================================================
class AutoFireOnReset(gym.Wrapper):
    """Automatically press FIRE after resetting."""

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
    """Convolutional network for estimating Q-values."""

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
    """Stores and samples experience for training."""

    def __init__(self, capacity: int):
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
# REWARD SHAPER - Focus on Survival and Timing
# ============================================================
class RewardShaper:
    """
    Reward shaping optimized for learning Pitfall timing patterns.

    Key principles:
    - Reward survival (longer episodes = more timing practice)
    - Reward exploration (both directions equally)
    - Reward action diversity (prevent collapse)
    - Death is feedback, not catastrophe
    """

    def __init__(self):
        self.prev_lives = None
        self.step_idx = 0
        self.episode_steps = 0  # Steps in current episode
        self._prev_room = None
        self._room_visit_count = {}
        self._prev_x_pos = None
        self._treasure_count = 0

        # Track exploration in BOTH directions
        self._max_x_right = 0
        self._max_x_left = 0
        self._initial_x = None

        # Idle detection
        self._idle_steps = 0
        self._steps_since_any_movement = 0

        # Action diversity tracking
        self._recent_actions = []
        self._action_window = 20

        # Survival tracking
        self._last_milestone = 0

    def reset(self):
        """Reset for new episode."""
        self.prev_lives = None
        self.episode_steps = 0
        self._prev_room = None
        self._room_visit_count = {}
        self._prev_x_pos = None
        self._treasure_count = 0
        self._max_x_right = 0
        self._max_x_left = 0
        self._initial_x = None
        self._idle_steps = 0
        self._steps_since_any_movement = 0
        self._recent_actions = []
        self._last_milestone = 0

    def shape(self, env_info: dict, base_reward: float, action: int) -> tuple:
        """
        Return (shaped_reward, flags).

        Rewards:
        - Survival: Stay alive longer to learn timing
        - Movement: Explore in any direction
        - Diversity: Use different actions
        - Exploration: Visit new screens
        """
        shaped = float(base_reward)
        flags = {
            "new_room": False,
            "treasure": False,
            "moved": False,
            "idle": False,
            "new_territory": False,
            "diverse_actions": False,
            "survival_milestone": False
        }

        self.episode_steps += 1

        # Track action for diversity
        self._recent_actions.append(action)
        if len(self._recent_actions) > self._action_window:
            self._recent_actions.pop(0)

        # Life penalty - death matters but isn't catastrophic
        lives = env_info.get("lives", env_info.get("ale.lives"))
        if USE_LIFE_PENALTY:
            if (self.prev_lives is not None) and (lives is not None) and (lives < self.prev_lives):
                shaped += LIFE_PENALTY
        if lives is not None:
            self.prev_lives = lives

        # Survival bonus - KEY for learning timing
        if USE_SURVIVAL_BONUS:
            shaped += SURVIVAL_BONUS  # Small per-step bonus

            # Milestone bonuses (every 500 steps)
            if self.episode_steps % SURVIVAL_MILESTONE_INTERVAL == 0 and self.episode_steps > self._last_milestone:
                shaped += SURVIVAL_MILESTONE_BONUS
                self._last_milestone = self.episode_steps
                flags["survival_milestone"] = True

        # Constant time penalty (simulates 20-min timer)
        if USE_TIME_PENALTY:
            shaped += TIME_PENALTY

        # Track position for movement rewards
        x_pos = env_info.get("x_pos", env_info.get("ale.x_pos"))

        if self._initial_x is None and x_pos is not None:
            self._initial_x = x_pos
            self._max_x_right = x_pos
            self._max_x_left = x_pos

        # Movement bonus - reward ANY horizontal movement
        if USE_MOVEMENT_BONUS and x_pos is not None and self._prev_x_pos is not None:
            x_delta = abs(x_pos - self._prev_x_pos)

            if x_delta >= MIN_MOVEMENT_THRESHOLD:
                # Agent is moving!
                movement_reward = MOVEMENT_BONUS * (x_delta / 5.0)
                shaped += movement_reward
                flags["moved"] = True
                self._steps_since_any_movement = 0
                self._idle_steps = 0

                # New max distance bonus (either direction)
                if USE_NEW_MAX_DISTANCE_BONUS:
                    if x_pos > self._max_x_right:
                        shaped += NEW_MAX_DISTANCE_BONUS
                        self._max_x_right = x_pos
                        flags["new_territory"] = True
                    elif x_pos < self._max_x_left:
                        shaped += NEW_MAX_DISTANCE_BONUS
                        self._max_x_left = x_pos
                        flags["new_territory"] = True
            else:
                # Standing still
                self._idle_steps += 1
                self._steps_since_any_movement += 1

                if USE_IDLE_PENALTY:
                    shaped += IDLE_PENALTY
                    flags["idle"] = True

        if x_pos is not None:
            self._prev_x_pos = x_pos

        # Extra penalty for being stuck too long
        if self._steps_since_any_movement > 50:
            shaped += -0.5

        # Action diversity bonus (prevents action collapse)
        if USE_ACTION_DIVERSITY_BONUS and len(self._recent_actions) >= self._action_window:
            unique_actions = len(set(self._recent_actions))
            if unique_actions >= 5:
                shaped += ACTION_DIVERSITY_BONUS * unique_actions
                flags["diverse_actions"] = True

        # Room exploration
        room = env_info.get("room", env_info.get("ale.room"))
        if USE_ROOM_BONUS and (room is not None):
            if (self._prev_room is not None) and (room != self._prev_room):
                self._room_visit_count[room] = self._room_visit_count.get(room, 0) + 1

                if self._room_visit_count[room] == 1:
                    shaped += ROOM_BONUS
                    flags["new_room"] = True
                elif self._room_visit_count[room] <= 3:
                    shaped += ROOM_BONUS * 0.2

        if room is not None:
            self._prev_room = room

        # Treasure collection - big bonus but not required
        if USE_TREASURE_BONUS and base_reward > 0:
            shaped += TREASURE_BONUS
            flags["treasure"] = True
            self._treasure_count += 1

        self.step_idx += 1
        return shaped, flags


# ============================================================
# DQN AGENT
# ============================================================
class DQNAgent:
    """Double DQN Agent optimized for long-term learning."""

    def __init__(self, state_shape, n_actions: int, save_path: Path, cfg: DQNConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = state_shape[-3:]

        self.actions_count = n_actions
        self.config = cfg
        self.step_count = 0
        self.memory = ReplayBuffer(cfg.replay_buffer_size)
        self.save_path = save_path

        self.policy_net = QNetwork(c, h, w, n_actions).to(self.device)
        self.target_net = QNetwork(c, h, w, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.scaler = torch.amp.GradScaler(enabled=USE_AMP and self.device.type == "cuda")

        self.last_q_max = None

    def current_epsilon(self):
        """Calculate current epsilon based on decay schedule."""
        progress = min(1.0, self.step_count / self.config.eps_decay_steps)
        return (1.0 - progress) * self.config.eps_start + progress * self.config.eps_end

    @torch.inference_mode()
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.current_epsilon():
            self.step_count += 1
            self.last_q_max = None
            return random.randrange(self.actions_count)

        state_tensor = torch.from_numpy(state).unsqueeze(0).float().div_(255.0).to(self.device)
        q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        self.last_q_max = float(q_values.max().item())
        self.step_count += 1
        return action

    def learn(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < max(self.config.warmup_steps, self.config.batch_size):
            return None
        if self.step_count % self.config.learn_every != 0:
            return None

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

        if self.step_count % self.config.save_interval == 0:
            self.save_checkpoint()

        return float(loss.item())

    def save_checkpoint(self):
        """Save model checkpoint."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        file = self.save_path / f"checkpoint_step_{self.step_count}.pth"
        torch.save({
            'step_count': self.step_count,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, file)
        print(f"💾 Saved checkpoint: {file.name}")

    def load_checkpoint(self, file: Path):
        """Load model checkpoint and resume training."""
        if not file.exists():
            print(f"⚠️  Checkpoint not found: {file}")
            return False

        checkpoint = torch.load(file, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
        print(f"✅ Loaded checkpoint from step {self.step_count}")
        print(f"   Current epsilon: {self.current_epsilon():.4f}")
        return True


# ============================================================
# LOGGER
# ============================================================
class MetricLogger:
    """Track and log training metrics."""

    def __init__(self, save_dir: Path):
        self.log_file = save_dir / "training_log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"{'Episode':>8}{'Step':>12}{'Epsilon':>10}{'EpReward':>12}{'EpLength':>12}"
                    f"{'AvgR@100':>12}{'AvgLen@100':>12}{'AvgLoss@100':>14}{'AvgQ@100':>12}{'Time':>20}\n")
        self.ep_rewards, self.ep_lengths, self.ep_losses = [], [], []
        self._reset_episode()
        self.ep_qs = []

    def _reset_episode(self):
        self.curr_reward = 0
        self.curr_length = 0
        self.curr_loss_sum = 0.0
        self.curr_loss_count = 0
        self.curr_q_sum = 0.0
        self.curr_q_count = 0

    def log_step(self, reward, loss, q_value):
        self.curr_reward += reward
        self.curr_length += 1
        if loss is not None:
            self.curr_loss_sum += loss
            self.curr_loss_count += 1
        if q_value is not None:
            self.curr_q_sum += float(q_value)
            self.curr_q_count += 1

    def log_episode(self):
        avg_loss = self.curr_loss_sum / self.curr_loss_count if self.curr_loss_count else 0.0
        avg_q = self.curr_q_sum / self.curr_q_count if self.curr_q_count else 0.0
        self.ep_rewards.append(self.curr_reward)
        self.ep_lengths.append(self.curr_length)
        self.ep_losses.append(avg_loss)
        self.ep_qs.append(avg_q)
        self._reset_episode()

    def record(self, episode, epsilon, step, episode_reward, episode_length):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mean_r = np.mean(self.ep_rewards[-100:])
        mean_len = np.mean(self.ep_lengths[-100:])
        mean_loss = np.mean(self.ep_losses[-100:])
        mean_q = np.mean(self.ep_qs[-100:])

        with open(self.log_file, "a") as f:
            f.write(f"{episode:8d}{step:12d}{epsilon:10.4f}{episode_reward:12.1f}{episode_length:12d}"
                    f"{mean_r:12.1f}{mean_len:12.1f}{mean_loss:14.6f}{mean_q:12.3f}{now:>20}\n")

        print(f"Ep {episode:5d} | Step {step:8d} | ε={epsilon:.4f} | "
              f"R={episode_reward:7.1f} | Len={episode_length:4d} | "
              f"R@100={mean_r:7.1f} | Len@100={mean_len:6.1f}")


# ============================================================
# ENVIRONMENT
# ============================================================
def create_env(render=False):
    """Create preprocessed Pitfall environment."""
    env = gym.make("ALE/Pitfall-v5",
                   render_mode=("human" if render else "rgb_array"),
                   frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                             frame_skip=FRAME_SKIP, scale_obs=False)
    env = FrameStack(env, num_stack=4)
    env = AutoFireOnReset(env)
    return env


# ============================================================
# TRAINING LOOP
# ============================================================
def train(resume_checkpoint: Path = None):
    """Main training loop."""
    set_global_seed(RANDOM_SEED)

    use_cuda = torch.cuda.is_available()
    print(f"🖥️  Device: {'CUDA' if use_cuda else 'CPU'}")

    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = CHECKPOINTS_DIR / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving to: {save_dir}")

    # Initialize
    logger = MetricLogger(save_dir)
    shaper = RewardShaper()
    env = create_env(render=RENDER_TRAINING)

    obs, _ = env.reset(seed=RANDOM_SEED)
    state = convert_obs_to_chw(obs)
    n_actions = env.action_space.n

    cfg = DQNConfig()
    agent = DQNAgent(state.shape, n_actions, save_dir, cfg)

    # Resume from checkpoint if provided
    if resume_checkpoint:
        if agent.load_checkpoint(resume_checkpoint):
            print(f"📊 Resuming training from step {agent.step_count}")
        else:
            print(f"⚠️  Starting fresh training")

    print(f"\n🎮 Training Configuration:")
    print(f"   Episodes: {NUM_EPISODES}")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Learning rate: {cfg.learning_rate}")
    print(f"   Replay buffer: {cfg.replay_buffer_size:,}")
    print(f"   Epsilon decay: {cfg.eps_start} → {cfg.eps_end} over {cfg.eps_decay_steps:,} steps")
    print(f"   Goal: Survive long, explore far, learn timing patterns\n")

    total_steps = 0

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        state = convert_obs_to_chw(obs)
        shaper.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Reward shaping (includes action for diversity tracking)
            shaped_reward, flags = shaper.shape(
                info if isinstance(info, dict) else {},
                reward,
                action
            )

            next_state = convert_obs_to_chw(next_obs)
            agent.memory.store(state, action, shaped_reward, next_state, done)

            loss = agent.learn()
            logger.log_step(shaped_reward, loss, agent.last_q_max)

            episode_reward += shaped_reward
            episode_length += 1
            state = next_state
            total_steps += 1

            # STRICT_MODE: stop at the 20-min frame budget
            if STRICT_MODE and (total_steps >= MAX_STEPS):
                done = True

        logger.log_episode()

        # Log every 10 episodes
        if episode % 10 == 0:
            logger.record(episode, agent.current_epsilon(), agent.step_count,
                          episode_reward, episode_length)

        # Progress milestones
        if episode % 100 == 0:
            avg_reward = np.mean(logger.ep_rewards[-100:])
            avg_length = np.mean(logger.ep_lengths[-100:])
            print(f"\n{'=' * 70}")
            print(f"📊 Milestone Report - Episode {episode}")
            print(f"{'=' * 70}")
            print(f"   Total steps: {agent.step_count:,}")
            print(f"   Epsilon: {agent.current_epsilon():.4f}")
            print(f"   Avg reward (last 100): {avg_reward:.1f}")
            print(f"   Avg length (last 100): {avg_length:.1f}")
            print(f"   Replay buffer: {len(agent.memory):,} / {cfg.replay_buffer_size:,}")

            # Give feedback on progress
            if avg_length > 1000:
                print(f"   🎉 Great survival! Episodes lasting 1000+ steps")
            elif avg_length > 500:
                print(f"   ✅ Good progress! Keep training for better timing")
            elif avg_length < 300:
                print(f"   ⚠️  Short episodes - agent still learning basics")

            if avg_reward > 0:
                print(f"   🌟 POSITIVE REWARDS! Agent is succeeding!")
            elif avg_reward > -50:
                print(f"   👍 Improving! Approaching positive rewards")

            print(f"{'=' * 70}\n")

    env.close()
    agent.save_checkpoint()  # Final save
    print(f"\n✅ Training complete!")
    print(f"   Final checkpoint: {save_dir}")
    print(f"   Total steps: {agent.step_count:,}")
    print(f"   Final epsilon: {agent.current_epsilon():.4f}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train Pitfall! RL Agent - Focus on survival and exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new training
  python train.py
  
  # Resume from checkpoint
  python train.py --resume checkpoints/2024-10-20_12-00-00/checkpoint_step_890000.pth
  
  # The agent will learn to:
  #   1. Move constantly (left and right exploration)
  #   2. Survive longer (rewards for staying alive)
  #   3. Time jumps over hazards (crocodiles, pits, logs)
  #   4. Use diverse actions (prevents getting stuck)
  
  # Expected timeline:
  #   100k steps: Basic movement and exploration
  #   500k steps: Avoiding some hazards consistently
  #   1M steps: Starting to time some jumps correctly
  #   1.5M steps: Good survival, exploring 1000+ steps per episode
  #   2M+ steps: Mastering timing on common hazards
        """
    )
    parser.add_argument("--resume", type=str,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    resume_path = Path(args.resume) if args.resume else None

    if resume_path:
        print(f"\n🔄 Resuming training from checkpoint:")
        print(f"   {resume_path}\n")
    else:
        print(f"\n🆕 Starting fresh training\n")

    train(resume_checkpoint=resume_path)


if __name__ == "__main__":
    main()
