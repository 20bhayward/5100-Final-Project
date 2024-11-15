# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

class TrainingVisualizer:
    def __init__(self, level_number=1):
        """Initialize the visualizer with the path to the training results."""
        self.level_number = level_number

        # Get the parent directory of where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)

        # Construct absolute paths
        csv_path = os.path.join(parent_dir, 'trainer', 'trained_agents', f'level{level_number}', 'training_results.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}. Make sure you've run the training for level {level_number} first.")

        self.df = pd.read_csv(csv_path)
        # Calculate additional metrics
        self.df['reward_ma'] = self.df['reward'].rolling(window=50).mean()
        self.df['steps_ma'] = self.df['steps'].rolling(window=50).mean()

        # Create level-specific plots directory using absolute path
        self.plots_dir = os.path.join(parent_dir, 'plots', f'level{level_number}_plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Could not find {csv_path}. Make sure you've run the training for level {level_number} first.")

            print(f"\nAttempting to read CSV file...")
            self.df = pd.read_csv(csv_path)
            print(f"Successfully read CSV file with {len(self.df)} rows")

            # Calculate additional metrics
            self.df['reward_ma'] = self.df['reward'].rolling(window=50).mean()
            self.df['steps_ma'] = self.df['steps'].rolling(window=50).mean()

            # Create level-specific plots directory
            self.plots_dir = f'plots/level{level_number}_plots'
            os.makedirs(self.plots_dir, exist_ok=True)
            print(f"Created plots directory at: {self.plots_dir}")

        except Exception as e:
            print(f"\nError during initialization: {str(e)}")
            raise

    def plot_reward_progress(self):
        """Plot and save reward progress over time."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.df['episode'], self.df['reward'], 'b-', alpha=0.3, label='Raw Reward')
        plt.plot(self.df['episode'], self.df['reward_ma'], 'r-', label='Moving Average (50 episodes)')
        plt.title(f'Reward Progress - Level {self.level_number}', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Reward', labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f'{self.plots_dir}/1_reward_progress.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Created {save_path}")
        plt.close()

    def plot_steps_progress(self):
        """Plot and save steps per episode."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.df['episode'], self.df['steps'], 'g-', alpha=0.3, label='Raw Steps')
        plt.plot(self.df['episode'], self.df['steps_ma'], 'r-', label='Moving Average (50 episodes)')
        plt.title(f'Steps per Episode - Level {self.level_number}', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Steps', labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f'{self.plots_dir}/2_steps_progress.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Created {save_path}")
        plt.close()

    def plot_epsilon_decay(self):
        """Plot and save epsilon decay."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.df['episode'], self.df['epsilon'], 'r-')
        plt.title(f'Exploration Rate (Epsilon) Decay - Level {self.level_number}', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Epsilon', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f'{self.plots_dir}/3_epsilon_decay.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Created {save_path}")
        plt.close()

    def plot_reward_distribution(self):
        """Plot and save reward distribution."""
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.df, x='reward', bins=50)
        plt.axvline(self.df['reward'].mean(), color='r', linestyle='--', label='Mean')
        plt.title(f'Reward Distribution - Level {self.level_number}', pad=20)
        plt.xlabel('Reward', labelpad=10)
        plt.ylabel('Count', labelpad=10)
        plt.legend()
        plt.tight_layout()
        save_path = f'{self.plots_dir}/4_reward_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Created {save_path}")
        plt.close()

    def plot_reward_vs_steps(self):
        """Plot and save reward vs steps relationship."""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.df['steps'], self.df['reward'],
                            alpha=0.5, c=self.df['episode'],
                            cmap='viridis')
        plt.colorbar(scatter, label='Episode')
        plt.title(f'Reward vs Steps Relationship - Level {self.level_number}', pad=20)
        plt.xlabel('Steps', labelpad=10)
        plt.ylabel('Reward', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f'{self.plots_dir}/5_reward_vs_steps.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Created {save_path}")
        plt.close()

    def plot_learning_progress(self):
        """Plot and save learning progress with error bars."""
        plt.figure(figsize=(12, 8))
        window_size = 100
        df_windows = self.df.groupby(self.df.index // window_size).agg({
            'reward': ['mean', 'std'],
            'steps': 'mean'
        })
        plt.errorbar(
            x=df_windows.index * window_size,
            y=df_windows[('reward', 'mean')],
            yerr=df_windows[('reward', 'std')],
            capsize=5,
            color='purple',
            ecolor='gray',
            alpha=0.7
        )
        plt.title(f'Learning Progress (per {window_size} episodes) - Level {self.level_number}', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Average Reward', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f'{self.plots_dir}/6_learning_progress.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Created {save_path}")
        plt.close()

    def generate_all_plots(self):
        """Generate all plots one by one."""
        print(f"\nGenerating plots for Level {self.level_number}...")
        self.plot_reward_progress()
        self.plot_steps_progress()
        self.plot_epsilon_decay()
        self.plot_reward_distribution()
        self.plot_reward_vs_steps()
        self.plot_learning_progress()
        print(f"\nAll plots have been saved in '{self.plots_dir}'")

    def print_training_summary(self):
        """Print a statistical summary of the training results."""
        print(f"\nTraining Summary Statistics - Level {self.level_number}")
        print("=" * 40)
        print(f"Total Episodes: {len(self.df)}")
        print(f"Average Reward: {self.df['reward'].mean():.2f} ± {self.df['reward'].std():.2f}")
        print(f"Best Reward: {self.df['reward'].max():.2f}")
        print(f"Average Steps per Episode: {self.df['steps'].mean():.2f}")
        print(f"Final Epsilon: {self.df['epsilon'].iloc[-1]:.3f}")

        first_100 = self.df.head(100)['reward'].mean()
        last_100 = self.df.tail(100)['reward'].mean()
        improvement = ((last_100 - first_100) / first_100) * 100 if first_100 != 0 else float('inf')

        print("\nLearning Progress")
        print("=" * 16)
        print(f"First 100 episodes average reward: {first_100:.2f}")
        print(f"Last 100 episodes average reward: {last_100:.2f}")
        print(f"Improvement: {improvement:.1f}%")

        success_threshold = 0
        success_rate = (self.df['reward'] > success_threshold).mean() * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Visualize training results for a specific level')
    parser.add_argument('--level', type=int, default=1, help='Level number to visualize (default: 1)')
    args = parser.parse_args()

    try:
        visualizer = TrainingVisualizer(level_number=args.level)
        visualizer.generate_all_plots()
        visualizer.print_training_summary()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo generate the CSV file:")
        print(f"1. Run the training first: python game.py --t --l {args.level}")
        print("2. Then run this visualization script with the same level number")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
