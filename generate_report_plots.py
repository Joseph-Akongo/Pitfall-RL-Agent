"""
generate_report_plots_refactored.py
-----------------------------------
Generate publication-quality plots from multiple training sessions.
Combines logs from different training runs that continued from checkpoints.

Usage:
  python generate_report_plots_refactored.py --logs log1.txt log2.txt log3.txt
  python generate_report_plots_refactored.py --log-dir path/to/logs/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure matplotlib for better fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_single_log(log_file: Path) -> dict:
    """Parse a single training log file."""
    episodes = []
    steps = []
    epsilons = []
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    avg_lengths = []
    avg_losses = []
    avg_qs = []

    print(f"  📄 Loading: {log_file.name}")

    with open(log_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            try:
                episodes.append(int(parts[0]))
                steps.append(int(parts[1]))
                epsilons.append(float(parts[2]))
                episode_rewards.append(float(parts[3]))
                episode_lengths.append(float(parts[4]))
                avg_rewards.append(float(parts[5]))
                avg_lengths.append(float(parts[6]))
                avg_losses.append(float(parts[7]))
                avg_qs.append(float(parts[8]))
            except (ValueError, IndexError):
                continue

    print(f"     ✓ Loaded {len(episodes)} episodes (steps {min(steps)}-{max(steps)})")

    return {
        "episodes": np.array(episodes),
        "steps": np.array(steps),
        "epsilons": np.array(epsilons),
        "episode_rewards": np.array(episode_rewards),
        "episode_lengths": np.array(episode_lengths),
        "avg_rewards": np.array(avg_rewards),
        "avg_lengths": np.array(avg_lengths),
        "avg_losses": np.array(avg_losses),
        "avg_qs": np.array(avg_qs),
        "source_file": log_file.name
    }


def combine_logs(log_files: list) -> dict:
    """
    Combine multiple training logs into a single continuous dataset.
    Assumes logs are from consecutive training sessions.
    """
    print("\n📊 Loading and combining training logs...")
    print("=" * 70)

    # Load all data first
    all_data = [load_single_log(f) for f in log_files]

    # Sort by starting step number (earliest training first)
    all_data = sorted(all_data, key=lambda d: d["steps"][0])

    if len(all_data) == 1:
        print("\n✓ Single log file loaded")
        return all_data[0]

    # Combine all data
    print(f"\n🔗 Combining {len(all_data)} training sessions...")

    combined = {
        "episodes": np.array([]),
        "steps": np.array([]),
        "epsilons": np.array([]),
        "episode_rewards": np.array([]),
        "episode_lengths": np.array([]),
        "avg_rewards": np.array([]),
        "avg_lengths": np.array([]),
        "avg_losses": np.array([]),
        "avg_qs": np.array([]),
        "session_boundaries": []  # Track where sessions change
    }

    cumulative_episodes = 0

    for i, data in enumerate(all_data):
        # Track session boundaries for potential annotations
        if i > 0:
            combined["session_boundaries"].append(cumulative_episodes)

        # Adjust episode numbers to be cumulative
        adjusted_episodes = data["episodes"] + cumulative_episodes

        # Append all data
        combined["episodes"] = np.concatenate([combined["episodes"], adjusted_episodes])
        combined["steps"] = np.concatenate([combined["steps"], data["steps"]])
        combined["epsilons"] = np.concatenate([combined["epsilons"], data["epsilons"]])
        combined["episode_rewards"] = np.concatenate([combined["episode_rewards"], data["episode_rewards"]])
        combined["episode_lengths"] = np.concatenate([combined["episode_lengths"], data["episode_lengths"]])
        combined["avg_rewards"] = np.concatenate([combined["avg_rewards"], data["avg_rewards"]])
        combined["avg_lengths"] = np.concatenate([combined["avg_lengths"], data["avg_lengths"]])
        combined["avg_losses"] = np.concatenate([combined["avg_losses"], data["avg_losses"]])
        combined["avg_qs"] = np.concatenate([combined["avg_qs"], data["avg_qs"]])

        cumulative_episodes = adjusted_episodes[-1]

    print(f"✓ Combined dataset: {len(combined['episodes'])} total episodes")
    print(f"  Total steps: {combined['steps'][0]:,} → {combined['steps'][-1]:,}")
    print(f"  Epsilon range: {combined['epsilons'][-1]:.4f} → {combined['epsilons'][0]:.4f}")
    print("=" * 70)

    return combined


def plot_figure1_training_curves(data: dict, output_dir: Path):
    """
    Figure 1: Main training curves showing complete learning progression.
    Two subplots: Average Reward and Average Episode Length.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Plot 1: Average Reward
    ax1.plot(data["episodes"], data["avg_rewards"],
             linewidth=2, color='#2E86AB', label='Avg Reward (100 eps)', alpha=0.8)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
    ax1.set_ylabel('Average Reward', fontweight='bold')
    ax1.set_title('Training Progress: Average Reward Over Time', fontweight='bold', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Annotate key points
    if len(data["avg_rewards"]) > 0:
        min_reward_idx = np.argmin(data["avg_rewards"])
        max_reward_idx = np.argmax(data["avg_rewards"])

        ax1.annotate(f'Lowest: {data["avg_rewards"][min_reward_idx]:.1f}',
                     xy=(data["episodes"][min_reward_idx], data["avg_rewards"][min_reward_idx]),
                     xytext=(10, -20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                     fontsize=9)

    # Mark session boundaries if multiple sessions
    if "session_boundaries" in data and data["session_boundaries"]:
        for boundary in data["session_boundaries"]:
            ax1.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # Plot 2: Average Episode Length
    ax2.plot(data["episodes"], data["avg_lengths"],
             linewidth=2, color='#A23B72', label='Avg Length (100 eps)', alpha=0.8)
    ax2.axhline(y=200, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline (~200)')
    ax2.set_xlabel('Episode Number', fontweight='bold')
    ax2.set_ylabel('Average Episode Length', fontweight='bold')
    ax2.set_title('Training Progress: Average Episode Length Over Time', fontweight='bold', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Annotate peak and improvement
    if len(data["avg_lengths"]) > 10:
        max_length_idx = np.argmax(data["avg_lengths"])
        baseline = 200  # Random baseline
        improvement = data["avg_lengths"][max_length_idx] / baseline

        ax2.annotate(f'Peak: {data["avg_lengths"][max_length_idx]:.0f} steps\n({improvement:.1f}× improvement)',
                     xy=(data["episodes"][max_length_idx], data["avg_lengths"][max_length_idx]),
                     xytext=(10, -35), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                     fontsize=9)

        # Annotate final performance
        final_length = data["avg_lengths"][-1]
        final_improvement = final_length / baseline
        ax2.annotate(f'Final: {final_length:.0f} steps\n({final_improvement:.1f}× baseline)',
                     xy=(data["episodes"][-1], final_length),
                     xytext=(-80, 20), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                     fontsize=9)

    # Mark session boundaries
    if "session_boundaries" in data and data["session_boundaries"]:
        for boundary in data["session_boundaries"]:
            ax2.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # Add training summary text
    total_steps = data["steps"][-1] - data["steps"][0]
    fig.text(0.5, 0.02,
             f'Total Training: {len(data["episodes"])} episodes | {total_steps/1e6:.2f}M steps | ε: {data["epsilons"][0]:.3f}→{data["epsilons"][-1]:.3f}',
             ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_file = output_dir / "figure1_training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

    return fig


def plot_figure2_four_panel(data: dict, output_dir: Path):
    """
    Figure 2: Four-panel comprehensive view of complete training.
    Shows Reward, Length, Loss, and Epsilon/Q-values across all sessions.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Comprehensive Training Metrics - Complete Training History',
                 fontsize=15, fontweight='bold')

    # Panel 1: Average Reward
    ax1.plot(data["episodes"], data["avg_rewards"], linewidth=2, color='#2E86AB', alpha=0.8)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (100 eps)')
    ax1.set_title('(a) Learning Progress - Reward', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Mark session boundaries
    if "session_boundaries" in data and data["session_boundaries"]:
        for boundary in data["session_boundaries"]:
            ax1.axvline(x=boundary, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Panel 2: Average Episode Length
    ax2.plot(data["episodes"], data["avg_lengths"], linewidth=2, color='#A23B72', alpha=0.8)
    ax2.axhline(y=200, color='gray', linestyle='--', linewidth=1, alpha=0.3, label='Random baseline')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Length (100 eps)')
    ax2.set_title('(b) Survival Improvement', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if "session_boundaries" in data and data["session_boundaries"]:
        for boundary in data["session_boundaries"]:
            ax2.axvline(x=boundary, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Panel 3: Training Loss
    ax3.plot(data["episodes"], data["avg_losses"], linewidth=2, color='#F18F01', alpha=0.8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Loss (100 eps)')
    ax3.set_title('(c) Training Stability - Loss', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    if "session_boundaries" in data and data["session_boundaries"]:
        for boundary in data["session_boundaries"]:
            ax3.axvline(x=boundary, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Panel 4: Epsilon and Q-values (dual axis)
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(data["episodes"], data["avg_qs"],
                     linewidth=2, color='#06A77D', label='Avg Q-value', alpha=0.8)
    line2 = ax4_twin.plot(data["episodes"], data["epsilons"],
                          linewidth=2, color='#D62828', linestyle='--', label='Epsilon', alpha=0.8)

    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Q-value (100 eps)', color='#06A77D')
    ax4_twin.set_ylabel('Epsilon (ε)', color='#D62828')
    ax4.set_title('(d) Q-values & Exploration Rate', fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#06A77D')
    ax4_twin.tick_params(axis='y', labelcolor='#D62828')
    ax4.grid(True, alpha=0.3)

    if "session_boundaries" in data and data["session_boundaries"]:
        for boundary in data["session_boundaries"]:
            ax4.axvline(x=boundary, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = output_dir / "figure2_four_panel.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

    return fig


def plot_figure3_milestone_comparison(data: dict, output_dir: Path):
    """
    Figure 3: Performance at key training milestones across complete training.
    """
    # Find key milestones based on total steps
    total_steps = data["steps"][-1]
    milestones_steps = [
        100_000,
        500_000,
        1_000_000,
        total_steps
    ]

    milestone_data = []

    for target_steps in milestones_steps:
        # Find closest episode to target steps
        idx = np.argmin(np.abs(data["steps"] - target_steps))
        milestone_data.append({
            "steps": data["steps"][idx],
            "episode": data["episodes"][idx],
            "avg_reward": data["avg_rewards"][idx],
            "avg_length": data["avg_lengths"][idx],
            "epsilon": data["epsilons"][idx],
            "avg_q": data["avg_qs"][idx]
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Performance at Training Milestones', fontsize=14, fontweight='bold')

    labels = [f'{m["steps"]/1000:.0f}k steps' for m in milestone_data]
    x_pos = np.arange(len(labels))

    # Panel 1: Average Reward
    rewards = [m["avg_reward"] for m in milestone_data]
    colors1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(labels)))
    bars1 = ax1.bar(x_pos, rewards, color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Average Reward (100 eps)', fontweight='bold')
    ax1.set_title('(a) Average Reward Progression', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: Episode Length
    lengths = [m["avg_length"] for m in milestone_data]
    colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
    bars2 = ax2.bar(x_pos, lengths, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.axhline(y=200, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                label='Random baseline', zorder=0)
    ax2.set_ylabel('Average Episode Length (100 eps)', fontweight='bold')
    ax2.set_title('(b) Survival Time Progression', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars2, lengths):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = output_dir / "figure3_milestones.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

    return fig


def plot_figure4_training_summary(data: dict, output_dir: Path):
    """
    Figure 4: Single plot with dual x-axis (episodes and steps).
    Shows the relationship between episodes and steps.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary axis: episodes
    color = '#A23B72'
    ax1.set_xlabel('Episode Number', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Average Episode Length (100 eps)', color=color, fontweight='bold', fontsize=12)
    line1 = ax1.plot(data["episodes"], data["avg_lengths"],
                     color=color, linewidth=2.5, alpha=0.8, label='Episode Length')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=200, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: steps
    ax2 = ax1.twiny()
    ax2.set_xlabel('Training Steps (millions)', fontweight='bold', fontsize=12, color='#2E86AB')
    ax2.plot(data["steps"] / 1e6, data["avg_lengths"], alpha=0)  # Invisible plot to sync axes
    ax2.tick_params(axis='x', labelcolor='#2E86AB')

    ax1.set_title('Complete Training History: Episode Length vs Steps',
                  fontsize=14, fontweight='bold', pad=20)

    # Add annotations
    max_idx = np.argmax(data["avg_lengths"])
    ax1.annotate(f'Peak: {data["avg_lengths"][max_idx]:.0f} steps\nEpisode {data["episodes"][max_idx]:.0f}',
                 xy=(data["episodes"][max_idx], data["avg_lengths"][max_idx]),
                 xytext=(30, -40), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2),
                 fontsize=10, fontweight='bold')

    # Mark session boundaries
    if "session_boundaries" in data and data["session_boundaries"]:
        for i, boundary in enumerate(data["session_boundaries"]):
            ax1.axvline(x=boundary, color='red', linestyle=':', linewidth=2, alpha=0.6)
            ax1.text(boundary, ax1.get_ylim()[1] * 0.95, f'Session {i+2}',
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_file = output_dir / "figure4_training_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

    return fig


def generate_training_statistics(data: dict, output_dir: Path):
    """Generate a text file with training statistics."""
    stats_file = output_dir / "training_statistics.txt"

    with open(stats_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TRAINING STATISTICS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Episodes: {len(data['episodes'])}\n")
        f.write(f"Total Steps: {data['steps'][-1]:,}\n")
        f.write(f"Step Range: {data['steps'][0]:,} → {data['steps'][-1]:,}\n\n")

        f.write(f"Epsilon Decay: {data['epsilons'][0]:.4f} → {data['epsilons'][-1]:.4f}\n\n")

        f.write("Episode Length Statistics:\n")
        f.write(f"  Peak: {np.max(data['avg_lengths']):.1f} steps\n")
        f.write(f"  Final: {data['avg_lengths'][-1]:.1f} steps\n")
        f.write(f"  Mean: {np.mean(data['avg_lengths']):.1f} steps\n")
        f.write(f"  Improvement over baseline (200): {data['avg_lengths'][-1]/200:.1f}×\n\n")

        f.write("Reward Statistics:\n")
        f.write(f"  Peak: {np.max(data['avg_rewards']):.1f}\n")
        f.write(f"  Lowest: {np.min(data['avg_rewards']):.1f}\n")
        f.write(f"  Final: {data['avg_rewards'][-1]:.1f}\n")
        f.write(f"  Mean: {np.mean(data['avg_rewards']):.1f}\n\n")

        f.write("Q-value Statistics:\n")
        f.write(f"  Initial: {data['avg_qs'][0]:.2f}\n")
        f.write(f"  Final: {data['avg_qs'][-1]:.2f}\n")
        f.write(f"  Peak: {np.max(data['avg_qs']):.2f}\n\n")

        if "session_boundaries" in data and data["session_boundaries"]:
            f.write(f"Training Sessions: {len(data['session_boundaries']) + 1}\n")
            f.write(f"Session boundaries at episodes: {data['session_boundaries']}\n\n")

        f.write("=" * 70 + "\n")

    print(f"✅ Saved: {stats_file}")


def generate_all_plots(log_files: list, output_dir: Path):
    """Generate all figures for the report."""
    print("\n" + "=" * 70)
    print("GENERATING COMPLETE TRAINING REPORT FIGURES")
    print("=" * 70 + "\n")

    # Load and combine data
    data = combine_logs(log_files)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    print("\n📊 Generating figures...")
    print("=" * 70 + "\n")

    plot_figure1_training_curves(data, output_dir)
    plot_figure2_four_panel(data, output_dir)
    plot_figure3_milestone_comparison(data, output_dir)
    plot_figure4_training_summary(data, output_dir)

    # Generate statistics
    generate_training_statistics(data, output_dir)

    print("\n" + "=" * 70)
    print(f"✅ All figures saved to: {output_dir}")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. figure1_training_curves.png - Complete learning curves")
    print("  2. figure2_four_panel.png - Comprehensive metrics")
    print("  3. figure3_milestones.png - Performance at key milestones")
    print("  4. figure4_training_summary.png - Dual-axis summary view")
    print("  5. training_statistics.txt - Numerical summary")
    print("\nInsert these into your report with proper captions!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from multiple training log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specify multiple log files
  python generate_report_plots_refactored.py --logs log1.txt log2.txt log3.txt

  # Use all .txt files in a directory
  python generate_report_plots_refactored.py --log-dir logs/

  # Specify output directory
  python generate_report_plots_refactored.py --logs *.txt --output final_figures/
        """
    )
    parser.add_argument("--logs", nargs='+', type=str,
                       help="Paths to training log files")
    parser.add_argument("--log-dir", type=str,
                       help="Directory containing training log files (uses all .txt files)")
    parser.add_argument("--output", type=str, default="report_figures",
                       help="Output directory for figures")

    args = parser.parse_args()

    # Determine which log files to use
    if args.logs:
        log_files = [Path(f) for f in args.logs]
    elif args.log_dir:
        log_dir = Path(args.log_dir)
        log_files = sorted(log_dir.glob("*.txt"))
    else:
        print("❌ Error: Must specify either --logs or --log-dir")
        print("\nExamples:")
        print("  python generate_report_plots_refactored.py --logs log1.txt log2.txt log3.txt")
        print("  python generate_report_plots_refactored.py --log-dir logs/")
        return

    # Validate files exist
    missing_files = [f for f in log_files if not f.exists()]
    if missing_files:
        print("❌ Error: The following files do not exist:")
        for f in missing_files:
            print(f"  - {f}")
        return

    if not log_files:
        print("❌ Error: No log files found")
        return

    output_dir = Path(args.output)

    # Generate all plots
    generate_all_plots(log_files, output_dir)


if __name__ == "__main__":
    main()