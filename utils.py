import math
import random
import torch
import matplotlib.pyplot as plt
from datareader import NEW_CLASS_NAMES


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot dan simpan riwayat training/validasi untuk loss dan akurasi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training dan Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(epochs_range, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training dan Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nPlot disimpan sebagai 'training_history.png'")
    plt.show()


def visualize_random_val_predictions(model, val_loader, num_classes: int, count: int = 10, device=None):
    """
    Ambil beberapa gambar random dari validation set, lakukan inferensi, dan visualisasikan.
    - Binary (1 logit): tampilkan Pred, Prob (untuk kelas prediksi), dan GT dengan nama kelas.
    - Multi-label: tampilkan hingga 2 kelas prediksi (p>=0.5) beserta probabilitas; jika tidak ada, tampilkan top-1.
    """
    model.eval()
    val_dataset = getattr(val_loader, 'dataset', None)
    if val_dataset is None:
        print("Validation dataset tidak tersedia dari loader, melewati visualisasi prediksi.")
        return

    n_total = len(val_dataset)
    if n_total == 0:
        print("Validation dataset kosong, melewati visualisasi prediksi.")
        return

    k = min(count, n_total)
    indices = random.sample(range(n_total), k)

    # Kumpulkan batch gambar dan label ground truth
    images = []
    gt_labels = []
    for idx in indices:
        sample = val_dataset[idx]
        img_tensor = sample[0]
        label_tensor = sample[1] if len(sample) > 1 else None
        images.append(img_tensor)
        gt_labels.append(label_tensor)

    # Sesuaikan device dengan model (CPU/GPU)
    if device is None:
        device = next(model.parameters()).device
    batch = torch.stack(images, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.sigmoid(outputs)
        # Binary jika output berisi satu logit per sampel
        is_binary = (outputs.ndim == 2 and outputs.shape[1] == 1) or (outputs.ndim == 1)

    # Util untuk konversi gambar agar nyaman ditampilkan
    def to_display_img(t: torch.Tensor):
        t = t.detach().cpu()
        if t.ndim == 3:
            img = t.clone().permute(1, 2, 0)  # HWC
            img_min = float(img.min())
            img_max = float(img.max())
            img = (img - img_min) / (img_max - img_min + 1e-8)
            if img.shape[2] == 1:
                return img.squeeze(2), 'gray'
            return img, None
        elif t.ndim == 2:
            img_min = float(t.min())
            img_max = float(t.max())
            img = (t - img_min) / (img_max - img_min + 1e-8)
            return img, 'gray'
        else:
            return t, None

    cols = 5
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        img = images[i]
        disp_img, cmap = to_display_img(img)
        ax.imshow(disp_img, cmap=cmap)
        ax.axis('off')

        # Teks prediksi + ground truth
        p = probs[i].detach().cpu()
        pred_line = ''
        prob_line = ''
        gt_line = ''

        def format_gt(lbl):
            if lbl is None:
                return '-'
            if isinstance(lbl, torch.Tensor):
                l = lbl.detach().cpu().float().flatten()
            else:
                try:
                    l = torch.tensor(lbl).float().flatten()
                except Exception:
                    return '-'
            if is_binary:
                v = int((l.squeeze().item()) >= 0.5)
                return NEW_CLASS_NAMES.get(v, str(v))
            else:
                chosen = (l >= 0.5).nonzero(as_tuple=True)[0].tolist()
                if len(chosen) == 0:
                    return '[]'
                names = [NEW_CLASS_NAMES.get(ci, str(ci)) for ci in chosen]
                if len(names) > 3:
                    names = names[:3] + ['…']
                return '[' + ', '.join(names) + ']'

        if is_binary:
            p_scalar = float(p.squeeze().item())
            pred_idx = 1 if p_scalar >= 0.5 else 0
            pred_name = NEW_CLASS_NAMES.get(pred_idx, str(pred_idx))
            p_pred = p_scalar if pred_idx == 1 else (1 - p_scalar)
            pred_line = f"Pred: {pred_name}"
            prob_line = f"Prob: {p_pred:.2f}"
            gt_line = f"GT: {format_gt(gt_labels[i])}"
        else:
            p_vec = p.flatten()
            thresh = 0.5
            chosen = (p_vec >= thresh).nonzero(as_tuple=True)[0].tolist()
            if len(chosen) == 0:
                top1 = int(torch.argmax(p_vec).item())
                name = NEW_CLASS_NAMES.get(top1, str(top1))
                pred_line = f"Pred: {name}"
                prob_line = f"Prob: {p_vec[top1].item():.2f}"
            else:
                chosen_sorted = sorted(chosen, key=lambda ci: float(p_vec[ci].item()), reverse=True)
                name_parts = []
                prob_parts = []
                for ci in chosen_sorted[:2]:
                    name = NEW_CLASS_NAMES.get(ci, str(ci))
                    name_parts.append(name)
                    prob_parts.append(f"{p_vec[ci].item():.2f}")
                if len(chosen_sorted) > 2:
                    name_parts.append('…')
                    prob_parts.append('…')
                pred_line = "Pred: " + ", ".join(name_parts)
                prob_line = "Prob: " + ", ".join(prob_parts)
            gt_line = f"GT: {format_gt(gt_labels[i])}"

        # Tiga baris teks berwarna untuk keterbacaan
        y0 = 0.98
        dy = 0.10
        ax.text(0.02, y0, pred_line, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, color='tab:blue', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))
        ax.text(0.02, y0 - dy, prob_line, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, color='tab:orange', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))
        ax.text(0.02, y0 - 2*dy, gt_line, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, color='tab:green', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

    # Matikan axes kosong jika ada
    total_axes = rows * cols
    if total_axes > k:
        for j in range(k, total_axes):
            ax = axes[j] if rows == 1 else axes[divmod(j, cols)]
            if rows > 1:
                r, c = divmod(j, cols)
                ax = axes[r][c]
            else:
                ax = axes[j]
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('val_predictions.png', dpi=300, bbox_inches='tight')
    print("\nVisualisasi prediksi validation disimpan sebagai 'val_predictions.png'")
    plt.show()
