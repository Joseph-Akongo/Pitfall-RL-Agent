"""
config.py
---------
Centralized configuration for the Pitfall! RL agent.
Optimized to:
- Prevent action collapse
- Encourage bidirectional exploration
- Reward survival and timing learning
- Focus on "making it far" rather than perfect treasure collection
- **NEW: Enhanced to encourage jumping over obstacles**
"""

from dataclasses import dataclass
from pathlib import Path

# ENVIRONMENT SETTINGS
STRICT_MODE = False  # True = authentic Pitfall scoring and 20-min cap
FRAME_SKIP = 4
MAX_STEPS = 10_000_000  # 20 minutes at 60 FPS (only applies if STRICT_MODE=True)
RENDER_TRAINING = False  # Set to True to watch the agent train (much slower)
RENDER_EVALUATION = True  # Set to True to watch the agent play during eval

# TRAINING SETTINGS
NUM_EPISODES = 50_000
RANDOM_SEED = 42

# REWARD SHAPING SETTINGS - Optimized for Survival and Timing
USE_LIFE_PENALTY = True
USE_TIME_PENALTY = True
USE_ROOM_BONUS = True
USE_TREASURE_BONUS = True
USE_MOVEMENT_BONUS = True  # Reward ANY horizontal movement
USE_IDLE_PENALTY = True
USE_NEW_MAX_DISTANCE_BONUS = True  # Reward exploring new territory
USE_ACTION_DIVERSITY_BONUS = True  # Prevent action collapse
USE_SURVIVAL_BONUS = True  # Reward surviving longer (helps learn timing)

# Death penalties (REDUCED - we want agent to experiment with jumps)
LIFE_PENALTY = -1.0  # Reduced from -2.0 - dying while learning to jump is OK

# Movement rewards - encourage exploration in both directions
MOVEMENT_BONUS = 5.0  # Reward for moving horizontally (any direction)
NEW_MAX_DISTANCE_BONUS = 20.0  # Big bonus for reaching new max distance (either direction)
ACTION_DIVERSITY_BONUS = 0.2  # INCREASED from 0.1 - reward using varied actions (like jumping!)

# Survival rewards - KEY for learning timing
SURVIVAL_BONUS = 0.1  # INCREASED from 0.05 - reward surviving with good actions
SURVIVAL_MILESTONE_BONUS = 15.0  # INCREASED from 10.0 - bigger reward for sustained survival
SURVIVAL_MILESTONE_INTERVAL = 500  # Steps between milestone bonuses

# Penalties for inaction
IDLE_PENALTY = -1.5  # INCREASED from -1.0 - standing still is very bad in Pitfall
TIME_PENALTY = -0.2  # Gentle constant pressure (simulates 20-min timer)

# Exploration rewards
ROOM_BONUS = 1.0  # Encourage visiting new screens

# Ultimate goal (but not required for "making it far")
TREASURE_BONUS = 150.0  # Large reward for collecting treasures

# Movement thresholds
MIN_MOVEMENT_THRESHOLD = 2  # Pixels to count as movement (not idle)

# DQN HYPERPARAMETERS - Optimized for Long-Term Learning
@dataclass
class DQNConfig:
    """Hyperparameters tuned for learning timing patterns over millions of steps."""

    # Discount factor - high for long-term planning
    gamma: float = 0.99  # Standard DQN value, proven stable

    # Network updates
    sync_target_every: int = 10_000  # Steps between target network updates
    learn_every: int = 4  # Learn every N steps

    # Batch training
    batch_size: int = 32  # Standard batch size
    warmup_steps: int = 50_000  # Build diverse experience before learning
    learning_rate: float = 0.00025  # Standard DQN learning rate

    # Exploration schedule - EXTENDED for jump learning
    eps_start: float = 1.0  # Full random exploration at start
    eps_end: float = 0.25  # INCREASED from 0.15 - keep 25% exploration (jumping needs experimentation!)
    eps_decay_steps: int = 3_500_000  # INCREASED from 2M - give agent 3.5M steps to learn jump timing

    # Memory - Large buffer essential for rare timing successes
    replay_buffer_size: int = 1_000_000  # Store diverse experiences (including rare successes)

    # Checkpointing
    save_interval: int = 10_000  # Steps between checkpoint saves

# PATHS
BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# CUDA SETTINGS
USE_AMP = False  # Use automatic mixed precision (faster on modern GPUs)

# REWARD BALANCE EXPLANATION
