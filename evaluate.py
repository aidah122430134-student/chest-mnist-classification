# evaluate.py
"""
Comprehensive Model Evaluation Script
Menghasilkan:
1. Confusion Matrix
2. ROC Curve
3. Precision-Recall Curve
4. Prediction Distribution
5. Performance Metrics (Accuracy, Precision, Recall, F1-Score, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import DenseNet121
from tqdm import tqdm


def evaluate_model(model_path='best_model_densenet121.pth', batch_size=12):
    """
    Evaluate model dan generate semua metrics dan visualizations
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, val_loader, num_classes, in_channels = get_data_loaders(batch_size)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = DenseNet121(in_channels=in_channels, num_classes=num_classes, pretrained=False).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded! Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Evaluation mode
    model.eval()
    
    # Collect predictions and ground truth
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    print("\nRunning inference on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.cpu().numpy().flatten()
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions = (probabilities > 0.5).astype(int)
            
            all_labels.extend(labels)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    
    # Per-class metrics
    print(f"\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_predictions, 
        target_names=[NEW_CLASS_NAMES[0], NEW_CLASS_NAMES[1]],
        digits=4
    ))
    
    # Create comprehensive visualization
    create_evaluation_plots(
        all_labels, 
        all_predictions, 
        all_probabilities,
        accuracy,
        precision,
        recall,
        f1
    )
    
    print("\n" + "="*70)
    print("All evaluation plots saved successfully!")
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'labels': all_labels,
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }


def create_evaluation_plots(labels, predictions, probabilities, acc, prec, rec, f1):
    """
    Create comprehensive evaluation plots
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    plot_confusion_matrix(labels, predictions, ax1)
    
    # 2. ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    plot_roc_curve(labels, probabilities, ax2)
    
    # 3. Precision-Recall Curve
    ax3 = plt.subplot(2, 3, 3)
    plot_precision_recall_curve(labels, probabilities, ax3)
    
    # 4. Prediction Distribution
    ax4 = plt.subplot(2, 3, 4)
    plot_prediction_distribution(labels, probabilities, ax4)
    
    # 5. Performance Metrics Bar Chart
    ax5 = plt.subplot(2, 3, 5)
    plot_metrics_bar(acc, prec, rec, f1, ax5)
    
    # 6. Per-Class Accuracy
    ax6 = plt.subplot(2, 3, 6)
    plot_per_class_metrics(labels, predictions, ax6)
    
    plt.tight_layout()
    plt.savefig('comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: comprehensive_evaluation.png")
    plt.close()
    
    # Create individual high-resolution plots
    create_individual_plots(labels, predictions, probabilities)


def plot_confusion_matrix(labels, predictions, ax):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[NEW_CLASS_NAMES[0], NEW_CLASS_NAMES[1]],
                yticklabels=[NEW_CLASS_NAMES[0], NEW_CLASS_NAMES[1]],
                cbar_kws={'label': 'Count'})
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.7, f'({cm_normalized[i,j]*100:.1f}%)', 
                   ha='center', va='center', fontsize=9, color='gray')
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)


def plot_roc_curve(labels, probabilities, ax):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_precision_recall_curve(labels, probabilities, ax):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    avg_precision = average_precision_score(labels, probabilities)
    
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.axhline(y=labels.mean(), color='red', linestyle='--', 
               label=f'Baseline (Prevalence = {labels.mean():.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_prediction_distribution(labels, probabilities, ax):
    """Plot prediction probability distribution"""
    # Separate probabilities by true class
    prob_class_0 = probabilities[labels == 0]
    prob_class_1 = probabilities[labels == 1]
    
    # Plot histograms
    ax.hist(prob_class_0, bins=30, alpha=0.6, label=NEW_CLASS_NAMES[0], 
            color='skyblue', edgecolor='black')
    ax.hist(prob_class_1, bins=30, alpha=0.6, label=NEW_CLASS_NAMES[1], 
            color='salmon', edgecolor='black')
    
    # Add threshold line
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
               label='Decision Threshold (0.5)')
    
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_metrics_bar(accuracy, precision, recall, f1, ax):
    """Plot performance metrics as bar chart"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim([0, 105])
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 92%
    ax.axhline(y=92, color='green', linestyle='--', linewidth=2, 
               label='Target (92%)', alpha=0.7)
    ax.legend(fontsize=9)


def plot_per_class_metrics(labels, predictions, ax):
    """Plot per-class precision, recall, F1"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    x = np.arange(len(NEW_CLASS_NAMES.values()))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision * 100, width, label='Precision', 
                   color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, recall * 100, width, label='Recall', 
                   color='lightcoral', edgecolor='black')
    bars3 = ax.bar(x + width, f1 * 100, width, label='F1-Score', 
                   color='lightgreen', edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(NEW_CLASS_NAMES.values(), fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])


def create_individual_plots(labels, predictions, probabilities):
    """Create individual high-resolution plots"""
    
    # 1. Confusion Matrix (Large)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_confusion_matrix(labels, predictions, ax)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve (Large)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_roc_curve(labels, probabilities, ax)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: roc_curve.png")
    plt.close()
    
    # 3. Precision-Recall Curve (Large)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_precision_recall_curve(labels, probabilities, ax)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: precision_recall_curve.png")
    plt.close()
    
    # 4. Prediction Distribution (Large)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_prediction_distribution(labels, probabilities, ax)
    plt.tight_layout()
    plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: prediction_distribution.png")
    plt.close()
    
    # 5. Detailed Metrics Table
    create_metrics_table(labels, predictions, probabilities)


def create_metrics_table(labels, predictions, probabilities):
    """Create detailed metrics table as image"""
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Overall metrics
    overall_precision = precision_score(labels, predictions)
    overall_recall = recall_score(labels, predictions)
    overall_f1 = f1_score(labels, predictions)
    
    # ROC AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(labels, probabilities)
    
    # Average Precision
    avg_precision = average_precision_score(labels, probabilities)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Data for table
    table_data = [
        ['Metric', NEW_CLASS_NAMES[0], NEW_CLASS_NAMES[1], 'Overall'],
        ['─' * 20, '─' * 15, '─' * 15, '─' * 15],
        ['Precision', f'{precision[0]*100:.2f}%', f'{precision[1]*100:.2f}%', f'{overall_precision*100:.2f}%'],
        ['Recall', f'{recall[0]*100:.2f}%', f'{recall[1]*100:.2f}%', f'{overall_recall*100:.2f}%'],
        ['F1-Score', f'{f1[0]*100:.2f}%', f'{f1[1]*100:.2f}%', f'{overall_f1*100:.2f}%'],
        ['Support', f'{int(support[0])}', f'{int(support[1])}', f'{int(support.sum())}'],
        ['─' * 20, '─' * 15, '─' * 15, '─' * 15],
        ['Overall Accuracy', '', '', f'{accuracy*100:.2f}%'],
        ['ROC AUC Score', '', '', f'{roc_auc:.4f}'],
        ['Avg Precision', '', '', f'{avg_precision:.4f}'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator rows
    for i in range(4):
        table[(1, i)].set_facecolor('#ecf0f1')
        table[(6, i)].set_facecolor('#ecf0f1')
    
    # Style overall metrics
    for i in range(4):
        table[(7, i)].set_facecolor('#e8f8f5')
        table[(8, i)].set_facecolor('#e8f8f5')
        table[(9, i)].set_facecolor('#e8f8f5')
    
    plt.title('Detailed Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('performance_metrics_table.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: performance_metrics_table.png")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("MODEL EVALUATION - COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Run evaluation
    results = evaluate_model()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. comprehensive_evaluation.png - All plots in one")
    print("  2. confusion_matrix.png - Confusion matrix")
    print("  3. roc_curve.png - ROC curve")
    print("  4. precision_recall_curve.png - Precision-Recall curve")
    print("  5. prediction_distribution.png - Prediction distribution")
    print("  6. performance_metrics_table.png - Detailed metrics table")
    print("\n" + "="*70)
