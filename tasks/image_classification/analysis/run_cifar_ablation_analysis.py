import torch
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from collections import defaultdict
from matplotlib.lines import Line2D

sns.set_style('darkgrid')
mpl.use('Agg')


def infer_experiment_group(checkpoint_path):
    """
    Infer experiment group from checkpoint path and return human-readable label.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        str: Human-readable experiment group name
    """
    path_lower = checkpoint_path.lower()
    
    # Map checkpoint path patterns to human-readable labels
    if "no-nlms" in path_lower:
        return "CTM (No NLMs)"
    elif "no-synch" in path_lower:
        return "CTM (No Synch)"
    elif "iters=50x15" in path_lower:
        return "CTM"
    else:
        return "Other"


def collect_checkpoint_files(root_directory):
    """
    Recursively collect all .pt checkpoint files from the given directory.
    
    Args:
        root_directory (str): Root directory to search for checkpoints
        
    Returns:
        list: Sorted list of checkpoint file paths
    """
    checkpoint_files = []
    
    for directory_path, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.pt'):
                full_path = os.path.join(directory_path, filename)
                checkpoint_files.append(full_path)
    
    return sorted(checkpoint_files)


def load_checkpoint_data(checkpoint_path, device='cpu'):
    """
    Load training metrics from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the checkpoint on
        
    Returns:
        dict: Dictionary containing training and test losses/accuracies
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    return {
        'train_losses': checkpoint.get('train_losses', []),
        'test_losses': checkpoint.get('test_losses', []),
        'train_accuracies': checkpoint.get('train_accuracies_most_certain', 
                                         checkpoint.get('train_accuracies', [])),
        'test_accuracies': checkpoint.get('test_accuracies_most_certain', 
                                        checkpoint.get('test_accuracies', []))
    }


def extract_final_metrics(metrics_data):
    """
    Extract final training metrics from loaded checkpoint data.
    
    Args:
        metrics_data (dict): Dictionary containing training metrics
        
    Returns:
        dict: Final values for each metric
    """
    def get_final_value(metric_list):
        if isinstance(metric_list, (list, np.ndarray)) and len(metric_list) > 0:
            return metric_list[-1]
        return np.nan
    
    return {
        'final_train_loss': get_final_value(metrics_data['train_losses']),
        'final_test_loss': get_final_value(metrics_data['test_losses']),
        'final_train_accuracy': get_final_value(metrics_data['train_accuracies']),
        'final_test_accuracy': get_final_value(metrics_data['test_accuracies'])
    }


def create_results_dataframe(checkpoint_files, device):
    """
    Process all checkpoint files and create a results DataFrame.
    
    Args:
        checkpoint_files (list): List of checkpoint file paths
        device (str): Device to load checkpoints on
        
    Returns:
        pd.DataFrame: DataFrame with results for each checkpoint
    """
    results = []
    
    for checkpoint_path in checkpoint_files:
        try:
            # Load checkpoint data
            metrics_data = load_checkpoint_data(checkpoint_path, device)
            final_metrics = extract_final_metrics(metrics_data)
            
            # Create record
            record = {
                'checkpoint_path': checkpoint_path,
                'experiment_group': infer_experiment_group(checkpoint_path),
                **final_metrics
            }
            results.append(record)
            
        except Exception as error:
            print(f"✗ Failed to process {checkpoint_path}: {error}")
    
    return pd.DataFrame(results)


def print_summary_statistics(results_df):
    """Print grouped summary statistics."""
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    
    metric_columns = ['final_train_accuracy', 'final_test_accuracy', 
                     'final_train_loss', 'final_test_loss']
    
    grouped_stats = results_df.groupby('experiment_group')[metric_columns].agg(['mean', 'std', 'count'])
    
    for group in grouped_stats.index:
        print(f"\n{group}:")
        print(f"  Train Accuracy: {grouped_stats.loc[group, ('final_train_accuracy', 'mean')]:.4f} ± {grouped_stats.loc[group, ('final_train_accuracy', 'std')]:.4f}")
        print(f"  Test Accuracy:  {grouped_stats.loc[group, ('final_test_accuracy', 'mean')]:.4f} ± {grouped_stats.loc[group, ('final_test_accuracy', 'std')]:.4f}")
        print(f"  Train Loss:     {grouped_stats.loc[group, ('final_train_loss', 'mean')]:.4f} ± {grouped_stats.loc[group, ('final_train_loss', 'std')]:.4f}")
        print(f"  Test Loss:      {grouped_stats.loc[group, ('final_test_loss', 'mean')]:.4f} ± {grouped_stats.loc[group, ('final_test_loss', 'std')]:.4f}")
        print(f"  Runs:           {grouped_stats.loc[group, ('final_train_accuracy', 'count')]}")


def plot_training_curves(results_df, device, output_dir="figures", step=1, scale=1.0, x_max=None, evaluate_every=1000):
    """
    Create and save averaged training curves for each experiment group with shaded std areas.
    Matches the style and dimensions of the reference plotting functions.
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        device (str): Device for loading checkpoints
        output_dir (str): Directory to save plots
        step (int): Step size for downsampling curves
        scale (float): Scale factor for figure size
        x_max (int): Maximum x-axis value
        evaluate_every (int): Training iterations between evaluations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by experiment type
    grouped_data = defaultdict(list)
    
    for group_name in results_df['experiment_group'].unique():
        group_checkpoints = results_df[results_df['experiment_group'] == group_name]
        
        train_accuracy_curves = []
        test_accuracy_curves = []
        
        # Collect curves from all runs in this group
        for _, row in group_checkpoints.iterrows():
            try:
                metrics_data = load_checkpoint_data(row['checkpoint_path'], device)
                
                train_acc = metrics_data['train_accuracies']
                test_acc = metrics_data['test_accuracies']
                
                if isinstance(train_acc, (list, np.ndarray)) and len(train_acc) > 0:
                    train_accuracy_curves.append(np.array(train_acc))
                if isinstance(test_acc, (list, np.ndarray)) and len(test_acc) > 0:
                    test_accuracy_curves.append(np.array(test_acc))
                    
            except Exception as error:
                print(f"✗ Failed to load curves from {row['checkpoint_path']}: {error}")
                continue
        
        grouped_data[group_name] = {
            'train_curves': train_accuracy_curves,
            'test_curves': test_accuracy_curves
        }
    
    # Get unique experiment groups and assign colors
    unique_groups = list(grouped_data.keys())
    base_colors = sns.color_palette("hls", n_colors=len(unique_groups))
    color_lookup = {group: base_colors[i] for i, group in enumerate(unique_groups)}
    
    # Create separate plots for train and test accuracy
    for plot_type in ['train', 'test']:
        fig, ax = plt.subplots(figsize=(scale * 10, scale * 5))
        global_max_x = 0
        
        for group_name, data in grouped_data.items():
            curves = data[f'{plot_type}_curves']
            if not curves:
                continue
                
            color = color_lookup[group_name]
            
            # Trim to minimum length and downsample
            min_len = min(len(curve) for curve in curves)
            trimmed = np.array([curve[:min_len] for curve in curves])[:, ::step]
            
            # Convert to percentage for accuracy
            mean = np.mean(trimmed, axis=0) * 100
            std = np.std(trimmed, axis=0) * 100
            
            # Create x-axis in terms of actual training iterations
            # Each point represents evaluate_every iterations, starting from evaluate_every
            x = (np.arange(len(mean)) + 1) * step * evaluate_every
            
            global_max_x = max(global_max_x, x[-1] if len(x) > 0 else 0)
            
            ax.plot(x, mean, color=color, label=group_name, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=color)
        
        # Customize plot to match reference style
        ax.set_xlabel("Training Iterations", fontsize=14)
        ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.grid(True, alpha=0.5)
        ax.legend(loc="upper left", fontsize=12)
        
        # Set axis limits and ticks to match reference style
        if x_max is None:
            x_max = max(60000, global_max_x)  # Ensure we show at least to 60k
        
        ax.set_xlim([0, x_max])
        if plot_type == 'test':
            ax.set_ylim(top=100)
        
        # Use fewer, more readable x-axis ticks
        ax.set_xticks(np.arange(0, x_max + 1, 20000))
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout(pad=0.1)
        
        # Save plots
        save_path = os.path.join(output_dir, f"{plot_type}_accuracy_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path.replace("png", "pdf"), format='pdf')
        plt.close(fig)
        
        print(f"✓ {plot_type.capitalize()} accuracy curves saved to {save_path}")
        print(f"  Data range: 0 to {global_max_x} iterations ({len(trimmed[0]) if len(trimmed) > 0 else 0} points)")
    
    # Also create a loss plot if loss data is available
    plot_loss_curves(results_df, device, output_dir, step, scale, x_max, evaluate_every)


def plot_loss_curves(results_df, device, output_dir="figures", step=1, scale=1.0, x_max=None, evaluate_every=1000):
    """
    Create and save averaged loss curves for each experiment group.
    """
    # Group data by experiment type
    grouped_data = defaultdict(list)
    
    for group_name in results_df['experiment_group'].unique():
        group_checkpoints = results_df[results_df['experiment_group'] == group_name]
        
        train_loss_curves = []
        
        # Collect curves from all runs in this group
        for _, row in group_checkpoints.iterrows():
            try:
                metrics_data = load_checkpoint_data(row['checkpoint_path'], device)
                
                train_loss = metrics_data['train_losses']
                
                if isinstance(train_loss, (list, np.ndarray)) and len(train_loss) > 0:
                    train_loss_curves.append(np.array(train_loss))
                    
            except Exception as error:
                print(f"✗ Failed to load loss curves from {row['checkpoint_path']}: {error}")
                continue
        
        grouped_data[group_name] = train_loss_curves
    
    # Get unique experiment groups and assign colors
    unique_groups = list(grouped_data.keys())
    base_colors = sns.color_palette("hls", n_colors=len(unique_groups))
    color_lookup = {group: base_colors[i] for i, group in enumerate(unique_groups)}
    
    fig, ax = plt.subplots(figsize=(scale * 10, scale * 5))
    global_max_x = 0
    
    for group_name, curves in grouped_data.items():
        if not curves:
            continue
            
        color = color_lookup[group_name]
        
        # Trim to minimum length and downsample
        min_len = min(len(curve) for curve in curves)
        trimmed = np.array([curve[:min_len] for curve in curves])[:, ::step]
        
        mean = np.mean(trimmed, axis=0)
        std = np.std(trimmed, axis=0)
        
        # Create x-axis in terms of actual training iterations
        # Each point represents evaluate_every iterations, starting from evaluate_every
        x = (np.arange(len(mean)) + 1) * step * evaluate_every
        
        global_max_x = max(global_max_x, x[-1] if len(x) > 0 else 0)
        
        ax.plot(x, mean, color=color, label=group_name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=color)
    
    # Customize plot to match reference style
    ax.set_xlabel("Training Iterations", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper left", fontsize=12)
    
    # Set axis limits and ticks to match reference style
    if x_max is None:
        x_max = max(60000, global_max_x)  # Ensure we show at least to 60k
    
    ax.set_xlim([0, x_max])
    ax.set_ylim(bottom=0)
    
    # Use fewer, more readable x-axis ticks
    ax.set_xticks(np.arange(0, x_max + 1, 40000))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout(pad=0.1)
    
    # Save plots
    save_path = os.path.join(output_dir, "train_loss_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace("png", "pdf"), format='pdf')
    plt.close(fig)
    
    print(f"✓ Training loss curves saved to {save_path}")
    print(f"  Data range: 0 to {global_max_x} iterations ({len(trimmed[0]) if len(trimmed) > 0 else 0} points)")


def setup_device(device_args):
    """Setup computation device based on arguments."""
    if device_args[0] != -1 and torch.cuda.is_available():
        device = f'cuda:{device_args[0]}'
        print(f"Using GPU: {device}")
    else:
        device = 'cpu'
        print(f"Using CPU")
    
    return device


def parse_command_line_arguments():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize Continuous Thought Machine training results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--device', 
        type=int, 
        nargs='+', 
        default=[-1], 
        help="GPU device index or -1 for CPU"
    )
    
    parser.add_argument(
        '--checkpoint_dirs', 
        type=str, 
        default='logs_backup_2/', 
        help="Path to directory containing CTM checkpoints"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures',
        help="Directory to save output plots"
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help="Step size for downsampling curves"
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help="Scale factor for figure size"
    )
    
    parser.add_argument(
        '--x_max',
        type=int,
        default=None,
        help="Maximum x-axis value for plots"
    )
    
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1000,
        help="Training iterations between evaluations"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments and setup
    args = parse_command_line_arguments()
    device = setup_device(args.device)
    
    print(f"Analyzing checkpoints from: {args.checkpoint_dirs}")
    
    # Collect and process checkpoint files
    checkpoint_files = collect_checkpoint_files(args.checkpoint_dirs)
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    if not checkpoint_files:
        print("No checkpoint files found. Please check the checkpoint directory.")
        return
    
    # Create results dataframe
    results_df = create_results_dataframe(checkpoint_files, device)
    
    if results_df.empty:
        print("No valid checkpoints could be processed.")
        return
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    # Generate and save plots
    plot_training_curves(results_df, device, args.output_dir, args.step, args.scale, args.x_max, args.evaluate_every)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()