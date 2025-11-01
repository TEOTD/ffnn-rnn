import json
import matplotlib.pyplot as plt
import sys

def plot_learning_curves(history_file):
    """Plot learning curves from saved history file"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, history['train_accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add best validation accuracy annotation
    best_epoch = history['val_accuracy'].index(max(history['val_accuracy'])) + 1
    best_acc = max(history['val_accuracy'])
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Val (Epoch {best_epoch})')
    ax2.text(best_epoch, best_acc, f' {best_acc:.3f}', fontsize=10, color='green')
    
    plt.tight_layout()
    
    # Save figure
    output_file = history_file.replace('.json', '.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        history_file = sys.argv[1]
    else:
        # Default to most recent FFNN history
        import glob
        files = glob.glob('ffnn_history_*.json')
        if not files:
            print("No history files found. Run training first.")
            sys.exit(1)
        history_file = max(files, key=lambda x: x)
    
    plot_learning_curves(history_file)
