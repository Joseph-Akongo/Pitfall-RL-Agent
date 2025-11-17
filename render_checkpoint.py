import argparse
import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from pathlib import Path
from collections import Counter

from config import FRAME_SKIP


ACTION_NAMES = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'UP',
    3: 'RIGHT',
    4: 'LEFT',
    5: 'DOWN',
    6: 'UPRIGHT',
    7: 'UPLEFT',
    8: 'DOWNRIGHT',
    9: 'DOWNLEFT',
}

def load_recording(filepath):
    """Load a recorded gameplay session."""
    data = np.load(filepath)
    actions = data['actions']
    timestamps = data.get('timestamps', None)
    
    return actions, timestamps


def print_recording_stats(actions):
    """Print statistics about the recording."""
    action_counts = Counter(actions)
    total = len(actions)

    # Jump analysis
    jump_actions = {2, 6, 7}
    jump_count = sum(action_counts.get(a, 0) for a in jump_actions)
    
    if jump_count > 0:
        jump_pct = 100 * jump_count / total
        diagonal = action_counts.get(6, 0) + action_counts.get(7, 0)
    
    # Movement analysis
    movement = action_counts.get(3, 0) + action_counts.get(4, 0)
    movement_pct = 100 * movement / total
    noop_pct = 100 * action_counts.get(0, 0) / total

def replay_recording(actions, timestamps, meta, display=True):

    repeat_p = float(meta.get("repeat_action_probability", 0.0))
    noop_max = int(meta.get("noop_max", 0))
    seed     = int(meta.get("seed", 12345))

    render_mode = "human" if display else "rgb_array"
    env = gym.make("ale_py:ALE/Pitfall-v5", frameskip=1, render_mode="human")  # or rgb_array

    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=False,
                             frame_skip=FRAME_SKIP, scale_obs=False, noop_max=noop_max)

    obs, info = env.reset(seed=seed)

    repeat_p = float(meta.get("repeat_action_probability", 0.0))
    noop_max = int(meta.get("noop_max", 0))
    seed = int(meta.get("seed", 12345))

    render_mode = "human" if display else "rgb_array"
    env = gym.make(
        "ALE/Pitfall-v5",
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=repeat_p
    )
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=False,
        frame_skip=FRAME_SKIP,
        scale_obs=False,
        noop_max=noop_max
    )

    obs, info = env.reset(seed=seed)
    
    step = 0
    total_reward = 0
    lives = 3
    prev_lives = 3

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        step += 1
        
        # Track lives
        curr_lives = info.get("lives", info.get("ale.lives", 3))
        if curr_lives < prev_lives:
            print(f"Life lost at step {step}! Lives: {curr_lives}")
            prev_lives = curr_lives
        
        # Progress
        if step % 50 == 0:
            x_pos = info.get("x_pos", info.get("ale.x_pos", 0))
            room = info.get("room", info.get("ale.room", 0))
            print(f"   Step {step:4d} | Score: {total_reward:6.0f} | "
                  f"X: {x_pos:4d} | Room: {room:3d} | Lives: {curr_lives}", end="\r")
        
        if terminated or truncated:
            print(f"\nReplay ended at step {step}")
            break
        
        if display:
            time.sleep(1/30)
    
    env.close()

def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded gameplay"
    )
    parser.add_argument("--file", type=str, required=True,
                        help="Path to recorded gameplay file (.npz)")
    parser.add_argument("--no-display", action="store_true",
                        help="Replay without display (faster)")
    
    args = parser.parse_args()
    
    filepath = Path(args.file)
    if not filepath.exists():
        # Try in outputs directory
        filepath = Path("/mnt/user-data/outputs") / args.file
        if not filepath.exists():
            print(f"❌ File not found: {args.file}")
            return
    
    # Load recording
    actions, timestamps = load_recording(filepath)
    # Show statistics
    print_recording_stats(actions)

    # AFTER loading actions,timestamps
    data = np.load(filepath, allow_pickle=True)
    meta = dict(data["meta"].item()) if "meta" in data else {}

    print_recording_stats(actions)
    input("Press ENTER to start")

    replay_recording(actions, timestamps, meta, display=not args.no_display)

if __name__ == "__main__":

    main()
