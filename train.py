# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import DenseNet121
from focal_loss import FocalLoss, WeightedBCEWithLogitsLoss
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
from tqdm import tqdm

# --- Hyperparameter Optimized untuk DenseNet121 - Target 92%+ ---
EPOCHS = 50  # Lebih banyak epochs dengan freeze/unfreeze strategy
BATCH_SIZE = 12  # Batch size optimal untuk GPU RTX 3050 + 224x224
LEARNING_RATE = 0.0001  # Learning rate awal
WEIGHT_DECAY = 5e-5  # L2 regularization
PATIENCE = 15  # Early stopping patience lebih panjang
LABEL_SMOOTHING = 0.1  # Label smoothing
USE_FOCAL_LOSS = True  # Gunakan Focal Loss untuk better performance
FREEZE_EPOCHS = 10  # Freeze backbone untuk N epochs pertama

#Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # Setup device (GPU jika tersedia, jika tidak gunakan CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Device yang digunakan: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*50}\n")
    
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model DenseNet121 dengan pre-trained weights dan pindahkan ke GPU
    print("Loading DenseNet121 with pre-trained ImageNet weights...")
    model = DenseNet121(in_channels=in_channels, num_classes=num_classes, pretrained=True).to(device)
    
    # FREEZE STRATEGY: Freeze backbone terlebih dahulu, train classifier only
    # Ini membuat model lebih stabil dan konvergen lebih baik
    if FREEZE_EPOCHS > 0:
        print(f"\n🔒 Freezing DenseNet backbone for first {FREEZE_EPOCHS} epochs...")
        for name, param in model.densenet.features.named_parameters():
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    # Calculate class weights untuk handling imbalance (Cardiomegaly:754 vs Pneumothorax:1552)
    pos_weight = torch.tensor([1552.0 / 754.0]).to(device)  # ratio of neg/pos
    
    if USE_FOCAL_LOSS:
        print("Using Focal Loss (better for medical imaging)")
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        print("Using Weighted BCE Loss")
        criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Menggunakan AdamW (Adam with weight decay) untuk better generalization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning Rate Scheduler - CosineAnnealingWarmRestarts untuk better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Gradient clipping untuk stabilitas
    max_grad_norm = 1.0
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    # Early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop dengan Freeze/Unfreeze Strategy
    for epoch in range(EPOCHS):
        # UNFREEZE backbone setelah FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS and FREEZE_EPOCHS > 0:
            print(f"\n🔓 Unfreezing DenseNet backbone at epoch {epoch+1}")
            print("Now fine-tuning the entire model...")
            for param in model.parameters():
                param.requires_grad = True
            
            # Update optimizer dengan learning rate lebih kecil untuk backbone
            optimizer = optim.AdamW([
                {'params': model.densenet.features.parameters(), 'lr': LEARNING_RATE * 0.1},  # Backbone: LR lebih kecil
                {'params': model.densenet.classifier.parameters(), 'lr': LEARNING_RATE}        # Classifier: LR normal
            ], weight_decay=WEIGHT_DECAY)
            
            # Reset scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters after unfreezing: {trainable_params:,}")
        
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar untuk training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        for images, labels in train_pbar:
            # Pindahkan data ke GPU
            images = images.to(device)
            # Ubah tipe data label menjadi float untuk BCEWithLogitsLoss
            labels = labels.float().to(device)
            
            # Apply label smoothing manually
            labels_smooth = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            
            outputs = model(images)
            loss = criterion(outputs, labels_smooth)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy (tetap pakai label asli, bukan smoothed)
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100 * train_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        # Progress bar untuk validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
        with torch.no_grad():
            for images, labels in val_pbar:
                # Pindahkan data ke GPU
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix({'loss': f'{val_loss.item():.4f}', 'acc': f'{current_val_acc:.2f}%'})
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate scheduler (CosineAnnealing updates every epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Cek jika model terbaik
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.2e} ⭐ BEST!")
        else:
            patience_counter += 1
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.2e}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Simpan model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
    }, 'best_model_densenet121.pth')
    print("Model terbaik disimpan sebagai 'best_model_densenet121.pth'")

    print("--- Training Selesai ---")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10, device=device)

if __name__ == '__main__':
    train()
    