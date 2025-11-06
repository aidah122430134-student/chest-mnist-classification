# model.py

import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet121(nn.Module):
    """
    DenseNet121 model untuk klasifikasi medical images.
    Menggunakan pre-trained weights dari ImageNet dan fine-tuning untuk ChestMNIST.
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Modifikasi layer pertama untuk menerima grayscale images (1 channel)
        if in_channels == 1:
            # DenseNet121 default menggunakan 3 channels (RGB)
            # Kita akan mengubahnya untuk menerima 1 channel (grayscale)
            original_conv = self.densenet.features.conv0
            self.densenet.features.conv0 = nn.Conv2d(
                in_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Jika menggunakan pretrained, rata-ratakan weights dari 3 channel ke 1 channel
            if pretrained:
                with torch.no_grad():
                    self.densenet.features.conv0.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        
        # Modifikasi classifier untuk binary classification
        num_features = self.densenet.classifier.in_features
        
        # Gunakan dropout untuk regularisasi
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1 if num_classes == 2 else num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)


# Backward compatibility - alias untuk SimpleCNN
SimpleCNN = DenseNet121

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'DenseNet121' ---")
    
    model = DenseNet121(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=False)
    print("Arsitektur Model:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with 224x224 input (standard for DenseNet)
    dummy_input = torch.randn(4, IN_CHANNELS, 224, 224)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'DenseNet121' berhasil!")

