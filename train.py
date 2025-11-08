"""
Ensemble Training untuk Target 92%+
Train 5 model dengan random seed berbeda, lalu ensemble predictions
Expected improvement: +0.5-1.0% dari single model

STANDALONE VERSION - No external dependencies on train scripts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import random
import math
import time
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.model_densenet import get_densenet_model
from data.datareader_highres import get_data_loaders

# Config (sama persis dengan V2 Improved yang berhasil 91.80%)
SEEDS = [42, 123, 456, 789, 2024]  # 5 different seeds
DROPOUT = 0.6
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
LR_FINETUNE = 0.00005
WEIGHT_DECAY = 0.01
STAGE1_EPOCHS = 25
STAGE2_EPOCHS = 100
PATIENCE = 35
GRAD_CLIP = 1.0
GRAD_ACCUMULATION = 4

def set_seed(seed):
    """Set all random seeds untuk reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accumulation=4, grad_clip=1.0):
    """Train for one epoch dengan gradient accumulation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).float()
        
        # Mixed precision training
        with autocast('cuda'):
            outputs = model(images).squeeze()  # Ensure [batch_size] shape
            loss = criterion(outputs, labels)
            loss = loss / grad_accumulation  # Scale loss
        
        # Backward
        scaler.scale(loss).backward()
        
        # Update weights setiap grad_accumulation steps
        if (i + 1) % grad_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * grad_accumulation
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate_with_tta(model, loader, criterion, device):
    """Validation dengan Test Time Augmentation (horizontal flip)"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            # Original prediction
            with autocast('cuda'):
                outputs1 = model(images).squeeze()
                loss = criterion(outputs1, labels)
            
            # Horizontal flip prediction (TTA)
            images_flipped = torch.flip(images, dims=[-1])
            with autocast('cuda'):
                outputs2 = model(images_flipped).squeeze()
            
            # Average predictions from original and flipped
            outputs_avg = (outputs1 + outputs2) / 2
            
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs_avg).cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Find optimal threshold
    best_acc = 0
    best_threshold = 0.5
    for threshold in np.arange(0.40, 0.61, 0.01):
        preds_binary = (all_preds > threshold).float()
        acc = (preds_binary == all_labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = best_acc * 100
    
    return epoch_loss, epoch_acc, best_threshold

def train_single_model(seed, save_path, device='cuda'):
    """Train single model dengan specific seed"""
    print(f"\n{'='*70}")
    print(f"Training Model with Seed: {seed}".center(70))
    print(f"{'='*70}\n")
    
    # Set seed
    set_seed(seed)
    
    # Get model
    model = get_densenet_model(model_type='standard', num_classes=1, dropout_rate=DROPOUT, freeze_backbone=False).to(device)
    
    # Get data
    train_loader, val_loader, _, _ = get_data_loaders(
        batch_size=BATCH_SIZE,
        use_augmentation=True
    )
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # Scaler for mixed precision
    scaler = GradScaler()
    
    # Best model tracking
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    print(f"📊 Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"⚙️  Config: Dropout={DROPOUT}, LR={LEARNING_RATE}, Batch={BATCH_SIZE}")
    print(f"🎯 Target: Break 92% barrier!\n")
    
    # ==================== STAGE 1: Train Classifier Only ====================
    print("="*70)
    print("STAGE 1: Training Classifier Only (Backbone Frozen)".center(70))
    print("="*70)
    
    # Freeze backbone
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                      patience=10)
    
    start_time = time.time()
    
    for epoch in range(1, STAGE1_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, 
                                                 scaler, device, GRAD_ACCUMULATION, GRAD_CLIP)
        val_loss, val_acc, threshold = validate_with_tta(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch:3d}/{STAGE1_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% | "
              f"Thresh: {threshold:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
    
    stage1_time = time.time() - start_time
    print(f"\nStage 1 completed in {stage1_time/60:.1f} minutes")
    print(f"Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # ==================== STAGE 2: Fine-tune Full Model ====================
    print("\n" + "="*70)
    print("STAGE 2: Full Model Fine-tuning".center(70))
    print("="*70)
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                      patience=10)
    
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, 
                                                 scaler, device, GRAD_ACCUMULATION, GRAD_CLIP)
        val_loss, val_acc, threshold = validate_with_tta(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {STAGE1_EPOCHS + epoch:3d}/{STAGE1_EPOCHS + STAGE2_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:6.2f}% | "
              f"Thresh: {threshold:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = STAGE1_EPOCHS + epoch
            patience_counter = 0
            
            # Create directory if not exists
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            
            # Save best model
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc / 100,
                'threshold': threshold,
                'seed': seed
            }, save_path)
            
            print(f"  💾 Model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered at epoch {STAGE1_EPOCHS + epoch}")
            break
    
    stage2_time = time.time() - start_time
    print(f"\nStage 2 completed in {stage2_time/60:.1f} minutes")
    print(f"Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Load best model
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_val_acc, best_epoch

def ensemble_predict(models, dataloader, device='cuda'):
    """Ensemble predictions dari multiple models dengan TTA"""
    all_preds = []
    all_labels = []
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Collect predictions from all models
            batch_preds = []
            for model in models:
                # Original
                pred1 = torch.sigmoid(model(images))
                # Horizontal flip (TTA)
                pred2 = torch.sigmoid(model(torch.flip(images, dims=[-1])))
                # Average TTA
                avg_pred = (pred1 + pred2) / 2
                batch_preds.append(avg_pred.cpu())
            
            # Average across all models
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            all_preds.append(ensemble_pred)
            all_labels.append(labels)
    
    return torch.cat(all_preds), torch.cat(all_labels)

def plot_ensemble_training_history(model_paths, seeds, individual_accs, output_path='results/ensemble_training_history.png'):
    """Plot training history dari semua model dalam ensemble"""
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Ensemble Training Results - 5 Models with Different Seeds', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Final Accuracy Bar Chart
    ax1 = axes[0]
    bars = ax1.bar(range(1, len(individual_accs) + 1), individual_accs, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    
    for i, (bar, acc) in enumerate(zip(bars, individual_accs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.axhline(y=92, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target: 92%')
    ax1.set_xlabel('Model ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Model Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(1, len(individual_accs) + 1))
    ax1.set_xticklabels([f'Model {i}\n(Seed {s})' for i, s in enumerate(seeds, 1)], fontsize=10)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(min(individual_accs) - 1, max(individual_accs) + 2)
    
    # Plot 2: Summary Statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    avg_acc = np.mean(individual_accs)
    std_acc = np.std(individual_accs)
    min_acc = np.min(individual_accs)
    max_acc = np.max(individual_accs)
    
    summary_text = f"""
    ENSEMBLE TRAINING SUMMARY
    ═══════════════════════════════════
    
    Configuration:
      • Architecture: DenseNet121
      • Dropout: {DROPOUT}
      • Batch Size: {BATCH_SIZE}
      • Learning Rate: {LEARNING_RATE} → {LR_FINETUNE}
      • Weight Decay: {WEIGHT_DECAY}
      • Gradient Accumulation: {GRAD_ACCUMULATION}
      • Patience: {PATIENCE} epochs
    
    Individual Model Results:
    """
    
    for i, (seed, acc) in enumerate(zip(seeds, individual_accs), 1):
        status = "✅" if acc >= 92 else "⚠️"
        summary_text += f"\n      {status} Model {i} (Seed {seed:4d}): {acc:5.2f}%"
    
    summary_text += f"""
    
    Statistics:
      • Average:   {avg_acc:.2f}%
      • Std Dev:   {std_acc:.2f}%
      • Min:       {min_acc:.2f}%
      • Max:       {max_acc:.2f}%
      • Range:     {max_acc - min_acc:.2f}%
    
    Target: 92.00%
    Models ≥ 92%: {sum(1 for acc in individual_accs if acc >= 92)}/5
    """
    
    ax2.text(0.05, 0.5, summary_text, transform=ax2.transAxes, 
            fontsize=10, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=15))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training history plot saved to: {output_path}")
    plt.close()


def visualize_ensemble_predictions(models, val_loader, num_samples=20, 
                                   output_path='results/ensemble_val_predictions.png'):
    """Visualize ensemble predictions on validation set"""
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get validation dataset
    val_dataset = val_loader.dataset
    n_total = len(val_dataset)
    
    # Sample random indices
    k = min(num_samples, n_total)
    indices = random.sample(range(n_total), k)
    
    # Collect images and labels
    images = []
    gt_labels = []
    for idx in indices:
        img_tensor, label = val_dataset[idx]
        images.append(img_tensor)
        gt_labels.append(label)
    
    batch = torch.stack(images).to(device)
    
    # Get ensemble predictions with TTA
    all_model_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            # Original
            pred1 = torch.sigmoid(model(batch))
            # Horizontal flip
            pred2 = torch.sigmoid(model(torch.flip(batch, dims=[-1])))
            # Average TTA
            avg_pred = (pred1 + pred2) / 2
            all_model_preds.append(avg_pred.cpu())
    
    # Ensemble predictions (average across models)
    ensemble_preds = torch.stack(all_model_preds).mean(dim=0).squeeze()
    
    # Class names
    class_names = {0: 'Cardiomegaly', 1: 'Pneumothorax'}
    
    # Plot
    cols = 5
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(f'Ensemble Predictions ({len(models)} Models + TTA)', 
                 fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(k):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        
        # Prepare image for display
        img = images[i].cpu().permute(1, 2, 0)
        if img.shape[2] == 1:
            img = img.squeeze(2)
            ax.imshow(img, cmap='gray')
        else:
            # Normalize for display
            img_min = img.min()
            img_max = img.max()
            img = (img - img_min) / (img_max - img_min + 1e-8)
            ax.imshow(img)
        
        ax.axis('off')
        
        # Get prediction
        prob = ensemble_preds[i].item()
        pred_class = 1 if prob >= 0.5 else 0
        gt_class = int(gt_labels[i].item())
        
        # Get confidence
        confidence = prob if pred_class == 1 else (1 - prob)
        
        # Prediction text
        pred_name = class_names[pred_class]
        gt_name = class_names[gt_class]
        
        # Color: green if correct, red if wrong
        is_correct = (pred_class == gt_class)
        text_color = 'green' if is_correct else 'red'
        check_mark = '✓' if is_correct else '✗'
        
        # Display text
        ax.text(0.02, 0.98, f"{check_mark} Pred: {pred_name}", 
               transform=ax.transAxes, va='top', ha='left',
               fontsize=10, color=text_color, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor=text_color, linewidth=2, pad=2))
        
        ax.text(0.02, 0.85, f"Conf: {confidence:.3f}", 
               transform=ax.transAxes, va='top', ha='left',
               fontsize=9, color='blue',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
        
        ax.text(0.02, 0.72, f"GT: {gt_name}", 
               transform=ax.transAxes, va='top', ha='left',
               fontsize=9, color='darkgreen',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
        
        # Show individual model agreement (variance indicator)
        individual_preds = [p[i].item() for p in all_model_preds]
        std = np.std(individual_preds)
        ax.text(0.02, 0.59, f"Std: {std:.3f}", 
               transform=ax.transAxes, va='top', ha='left',
               fontsize=8, color='purple',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
    
    # Hide empty subplots
    for i in range(k, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Validation predictions plot saved to: {output_path}")
    plt.close()

def find_optimal_threshold(predictions, labels):
    """Find optimal threshold untuk ensemble"""
    best_acc = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.40, 0.61, 0.01):
        preds_binary = (predictions > threshold).float()
        acc = (preds_binary.squeeze() == labels).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    return best_threshold, best_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*70)
    print("ENSEMBLE TRAINING FOR 92%+ ACCURACY".center(70))
    print("="*70)
    print(f"🚀 Device: {device}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    models = []
    model_paths = []
    individual_accs = []
    
    # Train 5 models
    print("\n" + "="*70)
    print("PHASE 1: Training 5 Models dengan Different Seeds".center(70))
    print("="*70)
    
    total_start_time = time.time()
    
    for i, seed in enumerate(SEEDS, 1):
        save_path = f"trained_models/ensemble_model_{i}_seed{seed}.pth"
        model_paths.append(save_path)
        
        print(f"\n{'🔄 TRAINING MODEL ' + str(i) + '/5':=^70}")
        model, val_acc, best_epoch = train_single_model(seed, save_path, device)
        models.append(model.to(device))
        individual_accs.append(val_acc)
        
        print(f"\n✅ Model {i} completed: {val_acc:.2f}% (Epoch {best_epoch})")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    total_train_time = time.time() - total_start_time
    
    # Load validation data
    print("\n" + "="*70)
    print("PHASE 2: Ensemble Evaluation".center(70))
    print("="*70)
    
    _, val_loader, _, _ = get_data_loaders(batch_size=32, use_augmentation=False)
    
    # Ensemble prediction
    print("\n🔮 Running ensemble predictions with TTA...")
    predictions, labels = ensemble_predict(models, val_loader, device)
    
    # Find optimal threshold
    print("🔍 Optimizing threshold...")
    optimal_threshold, ensemble_acc = find_optimal_threshold(predictions, labels)
    
    # Calculate metrics
    preds_binary = (predictions > optimal_threshold).squeeze()
    correct = (preds_binary == labels).sum().item()
    total = labels.size(0)
    
    # Per-model accuracies
    print("\n" + "="*70)
    print("📊 RESULTS".center(70))
    print("="*70)
    
    print("\n Individual Model Accuracies:")
    for i, (seed, acc) in enumerate(zip(SEEDS, individual_accs), 1):
        status = "✅" if acc >= 92 else "⚠️"
        print(f"   {status} Model {i} (Seed {seed:4d}): {acc:6.2f}%")
    
    print(f"\n✨ ENSEMBLE RESULTS:")
    print(f"   Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    print(f"   Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   Correct Predictions: {correct}/{total}")
    
    # Calculate improvement
    single_best = max(individual_accs)
    avg_single = np.mean(individual_accs)
    improvement = (ensemble_acc * 100) - single_best
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"   Best Single Model: {single_best:.2f}%")
    print(f"   Average Single: {avg_single:.2f}%")
    print(f"   Ensemble: {ensemble_acc*100:.2f}%")
    print(f"   Gain over Best: +{improvement:.2f}%")
    print(f"   Gain over Avg: +{(ensemble_acc*100) - avg_single:.2f}%")
    
    # Check if target reached
    if ensemble_acc >= 0.92:
        print("\n" + "🎉"*35)
        print("✅ TARGET 92% TERCAPAI!".center(70))
        print("🎉"*35)
    else:
        gap = (0.92 - ensemble_acc) * 100
        print(f"\n⚠️  Gap to 92%: {gap:.2f}%")
        print(f"💡 Try: More models, advanced TTA, or better architecture")
    
    print(f"\n⏱️  Total Training Time: {total_train_time/60:.1f} minutes")
    
    # Save ensemble predictions
    ensemble_data = {
        'model_paths': model_paths,
        'seeds': SEEDS,
        'individual_accuracies': individual_accs,
        'ensemble_accuracy': ensemble_acc,
        'optimal_threshold': optimal_threshold,
        'predictions': predictions,
        'labels': labels
    }
    
    save_path = Path('trained_models/ensemble_92plus.pth')
    torch.save(ensemble_data, save_path)
    print(f"\n💾 Ensemble data saved to: {save_path}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("PHASE 3: Generating Visualizations".center(70))
    print("="*70)
    
    print("\n📊 Generating training history plots...")
    plot_ensemble_training_history(model_paths, SEEDS, individual_accs)
    
    print("\n🖼️  Generating validation predictions visualizations...")
    visualize_ensemble_predictions(models, val_loader, num_samples=20)
    
    print("\n✅ All visualizations generated successfully!")
    print("   - results/ensemble_training_history.png")
    print("   - results/ensemble_val_predictions.png")
    
    print("\n" + "="*70)
    print("ENSEMBLE TRAINING COMPLETED!".center(70))
    print("="*70)

if __name__ == "__main__":
    main()
