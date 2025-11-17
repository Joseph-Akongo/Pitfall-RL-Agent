"""
evaluate.py
-----------
Load a trained checkpoint and watch the agent play Pitfall!
Usage: python evaluate.py --checkpoint PATH_TO_CHECKPOINT --episodes 5
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStack, AtariPreprocessing

from config import FRAME_SKIP, RENDER_EVALUATION
from train import QNetwork, AutoFireOnReset, convert_obs_to_chw


class EvaluationAgent:
    """Agent for evaluation (no exploration, just exploitation)."""

    def __init__(self, checkpoint_path: Path, n_actions: int, state_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = state_shape[-3:]

        self.policy_net = QNetwork(c, h, w, n_actions).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.policy_net.eval()

    def load_checkpoint(self, checkpoint_path: Path):
        """Load trained weights."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        print(f"✅ Loaded checkpoint from step {checkpoint['step_count']}")

    @torch.inference_mode()
    def choose_action(self, state):
        """Choose best action (greedy policy, no exploration)."""
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().div_(255.0).to(self.device)
        q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def create_eval_env():
    """Create environment for evaluation."""
    env = gym.make("ALE/Pitfall-v5",
                   render_mode=("human" if RENDER_EVALUATION else "rgb_array"),
                   frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                             frame_skip=FRAME_SKIP, scale_obs=False)
    env = FrameStack(env, num_stack=4)
    env = AutoFireOnReset(env)
    return env


def evaluate(checkpoint_path: Path, num_episodes: int = 5, max_steps: int = 18000):
    """Run evaluation episodes."""
    env = create_eval_env()

    # Get initial state to determine shape
    obs, _ = env.reset()
    state = convert_obs_to_chw(obs)
    n_actions = env.action_space.n

    # Create agent and load checkpoint
    agent = EvaluationAgent(checkpoint_path, n_actions, state.shape)

    print(f"\n🎮 Running {num_episodes} evaluation episodes...")
    print(f"{'Episode':>10}{'Score':>15}{'Steps':>15}{'Max Lives':>15}")
    print("-" * 55)

    episode_scores = []
    episode_steps = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = convert_obs_to_chw(obs)

        total_reward = 0
        steps = 0
        max_lives = 0
        done = False

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            obs, reward, terminated, truncated, info = env.step(action)

            # Track lives (if available)
            lives = info.get("lives", info.get("ale.lives", 0))
            if lives > max_lives:
                max_lives = lives

            done = terminated or truncated
            total_reward += reward
            state = convert_obs_to_chw(obs)
            steps += 1

            # Add small delay to make it easier to watch
            if RENDER_EVALUATION:
                time.sleep(0.01)

        episode_scores.append(total_reward)
        episode_steps.append(steps)

        print(f"{episode:10d}{total_reward:15.1f}{steps:15d}{max_lives:15d}")

    env.close()

    # Summary statistics
    print("-" * 55)
    print(f"{'Average':>10}{np.mean(episode_scores):15.1f}{np.mean(episode_steps):15.1f}")
    print(f"{'Std Dev':>10}{np.std(episode_scores):15.1f}{np.std(episode_steps):15.1f}")
    print(f"{'Max':>10}{np.max(episode_scores):15.1f}{np.max(episode_steps):15.1f}")
    print(f"{'Min':>10}{np.min(episode_scores):15.1f}{np.min(episode_steps):15.1f}")


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    """Find the most recent checkpoint."""
    checkpoint_folders = [f for f in checkpoints_dir.iterdir() if f.is_dir()]
    if not checkpoint_folders:
        raise FileNotFoundError(f"No checkpoint folders found in {checkpoints_dir}")

    # Get latest folder
    latest_folder = max(checkpoint_folders, key=lambda f: f.stat().st_mtime)

    # Find checkpoint with highest step count
    checkpoints = list(latest_folder.glob("checkpoint_step_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {latest_folder}")

    # Extract step number and get the latest
    latest_checkpoint = max(checkpoints,
                            key=lambda f: int(f.stem.split("_")[-1]))

    return latest_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Pitfall! agent")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint file (if not provided, uses latest)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--max-steps", type=int, default=18000,
                        help="Max steps per episode (default: 18000)")

    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        print("🔍 Looking for latest checkpoint...")
        from config import CHECKPOINTS_DIR
        checkpoint_path = find_latest_checkpoint(CHECKPOINTS_DIR)
        print(f"📁 Found: {checkpoint_path}")

    evaluate(checkpoint_path, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
