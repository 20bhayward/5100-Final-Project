import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class TrainingVisualizer:
    def __init__(self, csv_path='training_results.csv'):
        """Initialize the visualizer with the path to the training results."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}. Make sure you've run the training first.")

        self.df = pd.read_csv(csv_path)
        # Calculate additional metrics
        self.df['reward_ma'] = self.df['reward'].rolling(window=50).mean()
        self.df['steps_ma'] = self.df['steps'].rolling(window=50).mean()

        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)

    def plot_reward_progress(self):
        """Plot and save reward progress over time."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.df['episode'], self.df['reward'], 'b-', alpha=0.3, label='Raw Reward')
        plt.plot(self.df['episode'], self.df['reward_ma'], 'r-', label='Moving Average (50 episodes)')
        plt.title('Reward Progress', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Reward', labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/1_reward_progress.png', dpi=300, bbox_inches='tight')
        print("Created plots/1_reward_progress.png")
        plt.close()

    def plot_steps_progress(self):
        """Plot and save steps per episode."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.df['episode'], self.df['steps'], 'g-', alpha=0.3, label='Raw Steps')
        plt.plot(self.df['episode'], self.df['steps_ma'], 'r-', label='Moving Average (50 episodes)')
        plt.title('Steps per Episode', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Steps', labelpad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/2_steps_progress.png', dpi=300, bbox_inches='tight')
        print("Created plots/2_steps_progress.png")
        plt.close()

    def plot_epsilon_decay(self):
        """Plot and save epsilon decay."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.df['episode'], self.df['epsilon'], 'r-')
        plt.title('Exploration Rate (Epsilon) Decay', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Epsilon', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/3_epsilon_decay.png', dpi=300, bbox_inches='tight')
        print("Created plots/3_epsilon_decay.png")
        plt.close()

    def plot_reward_distribution(self):
        """Plot and save reward distribution."""
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.df, x='reward', bins=50)
        plt.axvline(self.df['reward'].mean(), color='r', linestyle='--', label='Mean')
        plt.title('Reward Distribution', pad=20)
        plt.xlabel('Reward', labelpad=10)
        plt.ylabel('Count', labelpad=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/4_reward_distribution.png', dpi=300, bbox_inches='tight')
        print("Created plots/4_reward_distribution.png")
        plt.close()

    def plot_reward_vs_steps(self):
        """Plot and save reward vs steps relationship."""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.df['steps'], self.df['reward'],
                            alpha=0.5, c=self.df['episode'],
                            cmap='viridis')
        plt.colorbar(scatter, label='Episode')
        plt.title('Reward vs Steps Relationship', pad=20)
        plt.xlabel('Steps', labelpad=10)
        plt.ylabel('Reward', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/5_reward_vs_steps.png', dpi=300, bbox_inches='tight')
        print("Created plots/5_reward_vs_steps.png")
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
        plt.title(f'Learning Progress (per {window_size} episodes)', pad=20)
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Average Reward', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/6_learning_progress.png', dpi=300, bbox_inches='tight')
        print("Created plots/6_learning_progress.png")
        plt.close()

    def generate_all_plots(self):
        """Generate all plots one by one."""
        print("\nGenerating individual plots...")
        self.plot_reward_progress()
        self.plot_steps_progress()
        self.plot_epsilon_decay()
        self.plot_reward_distribution()
        self.plot_reward_vs_steps()
        self.plot_learning_progress()
        print("\nAll plots have been saved in the 'plots' directory.")

    def print_training_summary(self):
        """Print a statistical summary of the training results."""
        print("\nTraining Summary Statistics")
        print("==========================")
        print(f"Total Episodes: {len(self.df)}")
        print(f"Average Reward: {self.df['reward'].mean():.2f} ± {self.df['reward'].std():.2f}")
        print(f"Best Reward: {self.df['reward'].max():.2f}")
        print(f"Average Steps per Episode: {self.df['steps'].mean():.2f}")
        print(f"Final Epsilon: {self.df['epsilon'].iloc[-1]:.3f}")

        # Calculate improvement
        first_100 = self.df.head(100)['reward'].mean()
        last_100 = self.df.tail(100)['reward'].mean()
        improvement = ((last_100 - first_100) / first_100) * 100 if first_100 != 0 else float('inf')

        print("\nLearning Progress")
        print("=================")
        print(f"First 100 episodes average reward: {first_100:.2f}")
        print(f"Last 100 episodes average reward: {last_100:.2f}")
        print(f"Improvement: {improvement:.1f}%")

        success_threshold = 0
        success_rate = (self.df['reward'] > success_threshold).mean() * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")

def main():
    try:
        visualizer = TrainingVisualizer()
        visualizer.generate_all_plots()
        visualizer.print_training_summary()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo generate the CSV file:")
        print("1. Run the training first: python game.py --t --l 1")
        print("2. Then run this visualization script")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
