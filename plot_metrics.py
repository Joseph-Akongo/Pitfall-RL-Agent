"""
plot_metrics.py
----------------
Automatically finds the most recent training log inside 'checkpoints/',
reads it, and plots the training reward and loss curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------------------
# Find the latest training log file
# -------------------------------------------------------------
def find_latest_log(checkpoints_folder: Path) -> Path:
    """Find the most recent checkpoints/<timestamp>/log.txt file."""
    # Get all subfolders under checkpoints/
    checkpoint_folders = [folder for folder in checkpoints_folder.iterdir() if folder.is_dir()]
    if not checkpoint_folders:
        raise FileNotFoundError(f"No subfolders found in: {checkpoints_folder}")

    # Pick the one with the newest modification time
    latest_folder = max(checkpoint_folders, key=lambda folder: folder.stat().st_mtime)

    log_file = latest_folder / "log.txt"
    if not log_file.exists():
        raise FileNotFoundError(f"log.txt not found in: {latest_folder}")

    return log_file


# -------------------------------------------------------------
# Load the training log data into numpy arrays
# -------------------------------------------------------------
def load_training_log(log_file: Path):
    """Read and parse the log.txt file written by MetricLogger."""
    episodes = []
    rewards = []
    losses = []

    with open(log_file, "r") as file:
        next(file)  # Skip the header line
        for line in file:
            if not line.strip():
                continue  # Skip blank lines

            # Extract numbers based on column positions
            episode_num = int(line[0:8])
            reward_avg = float(line[32:44])   # MeanR@100
            loss_avg = float(line[58:72])     # MeanLoss@100

            episodes.append(episode_num)
            rewards.append(reward_avg)
            losses.append(loss_avg)

    return {
        "episodes": np.array(episodes),
        "rewards": np.array(rewards),
        "losses": np.array(losses),
    }


# -------------------------------------------------------------
# Create and save the plots
# -------------------------------------------------------------
def plot_training_results(training_data, output_folder: Path):
    """Plot reward and loss curves, save as PNGs, and show them."""
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(training_data["episodes"], training_data["rewards"], label="Average Reward (last 100 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress - Average Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    reward_plot = output_folder / "reward_plot.png"
    plt.savefig(reward_plot, dpi=160)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(training_data["episodes"], training_data["losses"], label="Average Loss (last 100 episodes)", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Progress - Average Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot = output_folder / "loss_plot.png"
    plt.savefig(loss_plot, dpi=160)

    # Show both figures (in PyCharm or IDE)
    plt.show(block=True)
    print(f"✅ Plots saved to: {output_folder}")
    print(f"📈 Reward plot: {reward_plot}")
    print(f"📉 Loss plot:   {loss_plot}")


# -------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------
def main():
    base_folder = Path(__file__).resolve().parent
    checkpoints_folder = base_folder / "checkpoints"

    # Automatically find the newest log file
    latest_log = find_latest_log(checkpoints_folder)
    print(f"Found latest log file: {latest_log}")

    # Load and plot the data
    training_data = load_training_log(latest_log)
    plot_training_results(training_data, latest_log.parent)


if __name__ == "__main__":
    main()
