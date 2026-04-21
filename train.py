import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import time

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 384
BATCH_SIZE  = 8  # Reduced for EfficientNet-B7 (larger model)
EPOCHS      = 50
PATIENCE    = 10
GRAD_CLIP   = 1.0

# EfficientNet-B7 encoder - USE CORRECT TIMM NAME
ENCODER_NAME = "tf_efficientnet_b7"  # Correct name for EfficientNet-B7 in timm

MODEL_PATH      = f"best_model_{ENCODER_NAME}.pth"
LAST_MODEL_PATH = f"last_model_{ENCODER_NAME}.pth"
METRICS_CSV     = f"training_metrics_{ENCODER_NAME}.csv"
CROP_CACHE      = "ecc_crop_cache.json"

BASE_PATH       = '/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/segmentation_task'
TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train', 'images')
TRAIN_MASK_DIR  = os.path.join(BASE_PATH, 'train', 'masks')
TEST_IMAGE_DIR  = os.path.join(BASE_PATH, 'test',  'images')
TEST_MASK_DIR   = os.path.join(BASE_PATH, 'test',  'masks')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
ECC_PADDING   = 10

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_base_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def match_pairs(image_files, mask_files):
    mask_dict = {get_base_name(f): f for f in mask_files}
    pairs = [
        (img, mask_dict[get_base_name(img)])
        for img in image_files
        if get_base_name(img) in mask_dict
    ]
    if not pairs:
        raise ValueError("No matching image-mask pairs found.")
    imgs, masks = zip(*pairs)
    return list(imgs), list(masks)

def list_images(directory):
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

def list_masks(directory):
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith('.png')
    ])

# ─────────────────────────────────────────────
# ECC crop cache
# ─────────────────────────────────────────────
def compute_ecc_cache(mask_paths, cache_path, padding=ECC_PADDING):
    if os.path.exists(cache_path):
        print(f"Loading ECC cache from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print("Pre-computing ECC bounding boxes…")
    cache = {}
    for path in tqdm(mask_paths, desc="ECC cache"):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask_bin = ((mask > 0) * 1).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            lc = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(lc)
            cache[path] = [
                max(x - padding, 0), max(y - padding, 0),
                min(x + w + padding, mask.shape[1]),
                min(y + h + padding, mask.shape[0])
            ]
        else:
            cache[path] = None
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"ECC cache saved → {cache_path}")
    return cache

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class BRISCDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, crop_cache=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform
        self.crop_cache  = crop_cache or {}

    def _apply_crop(self, image, mask, mask_path):
        bbox = self.crop_cache.get(mask_path)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]
            mask  = mask[y1:y2,  x1:x2]
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 0).astype(np.float32)

        image, mask = self._apply_crop(image, mask, self.mask_paths[idx])
        image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.unsqueeze(0)

# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# File discovery & pairing
# ─────────────────────────────────────────────
train_image_files = list_images(TRAIN_IMAGE_DIR)
train_mask_files  = list_masks(TRAIN_MASK_DIR)
test_image_files  = list_images(TEST_IMAGE_DIR)
test_mask_files   = list_masks(TEST_MASK_DIR)

train_image_files, train_mask_files = match_pairs(train_image_files, train_mask_files)
test_image_files,  test_mask_files  = match_pairs(test_image_files,  test_mask_files)

print(f"Found {len(train_image_files)} training pairs")
print(f"Found {len(test_image_files)} test pairs")

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    train_image_files, train_mask_files, test_size=0.2, random_state=42
)

all_mask_paths = train_masks + val_masks + test_mask_files
crop_cache = compute_ecc_cache(all_mask_paths, CROP_CACHE)

train_dataset = BRISCDataset(train_imgs, train_masks, train_transform, crop_cache)
val_dataset   = BRISCDataset(val_imgs,   val_masks,   val_test_transform, crop_cache)
test_dataset  = BRISCDataset(test_image_files, test_mask_files, val_test_transform, crop_cache)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

# ─────────────────────────────────────────────
# Model: EfficientNet-B7 + UNet++ decoder
# ─────────────────────────────────────────────
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))

class cSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 4), in_channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sse = sSE(in_channels)
        self.cse = cSE(in_channels)
    def forward(self, x):
        return self.sse(x) + self.cse(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = scSE(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.attention(x)

class EfficientNetUNetPlusPlus(nn.Module):
    """
    EfficientNet-B7 encoder + UNet++ decoder with scSE attention.
    Using tf_efficientnet_b7 which has feature channels: [48, 80, 224, 640]
    """
    
    def __init__(self, encoder_name="tf_efficientnet_b7", pretrained=True,
                 decoder_channels=(512, 256, 128, 64), num_classes=1):
        super().__init__()
        
        # Encoder - EfficientNet-B7
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # Get 4 feature maps from stages 2-5
        )
        
        # Infer encoder output channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feats = self.encoder(dummy)
            enc_channels = [f.shape[1] for f in feats]
        
        print(f"Encoder: {encoder_name}")
        print(f"Feature channels: {enc_channels}")
        
        # Decoder (UNet++ style)
        self.decoder4 = DecoderBlock(enc_channels[3], enc_channels[2], decoder_channels[0])
        self.decoder3 = DecoderBlock(decoder_channels[0], enc_channels[1], decoder_channels[1])
        self.decoder2 = DecoderBlock(decoder_channels[1], enc_channels[0], decoder_channels[2])
        self.decoder1 = DecoderBlock(decoder_channels[2], 0, decoder_channels[3])
        
        # Segmentation head
        self.head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
    
    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        
        d4 = self.decoder4(f4, f3)
        d3 = self.decoder3(d4, f2)
        d2 = self.decoder2(d3, f1)
        d1 = self.decoder1(d2)
        
        out = F.interpolate(self.head(d1), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

# Initialize model
model = EfficientNetUNetPlusPlus(
    encoder_name=ENCODER_NAME,
    pretrained=True,
    decoder_channels=(512, 256, 128, 64),
    num_classes=1,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target)

# ─────────────────────────────────────────────
# Metric accumulator
# ─────────────────────────────────────────────
class MetricAccumulator:
    def __init__(self):
        self.reset()
    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0
    def update(self, pred_logits, target):
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        target = target.float()
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        self.tn += ((1 - pred) * (1 - target)).sum().item()
    @property
    def dice(self):
        return (2 * self.tp + 1e-7) / (2 * self.tp + self.fp + self.fn + 1e-7)
    @property
    def iou(self):
        return (self.tp + 1e-7) / (self.tp + self.fp + self.fn + 1e-7)
    @property
    def precision(self):
        return (self.tp + 1e-7) / (self.tp + self.fp + 1e-7)
    @property
    def recall(self):
        return (self.tp + 1e-7) / (self.tp + self.fn + 1e-7)
    @property
    def specificity(self):
        return (self.tn + 1e-7) / (self.tn + self.fp + 1e-7)
    def summary(self):
        return {
            'dice': self.dice,
            'iou': self.iou,
            'precision': self.precision,
            'recall': self.recall,
            'specificity': self.specificity,
        }

# ─────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────
def build_optimizer(model, base_lr=1e-4, encoder_lr_scale=0.1, weight_decay=1e-5):
    encoder_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_params if 'encoder' in name else decoder_params).append(param)
    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': base_lr * encoder_lr_scale},
        {'params': decoder_params, 'lr': base_lr},
    ], weight_decay=weight_decay)

optimizer = build_optimizer(model, base_lr=1e-4, encoder_lr_scale=0.1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 1e-4],
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
)
scaler = torch.cuda.amp.GradScaler()

# ─────────────────────────────────────────────
# Training state
# ─────────────────────────────────────────────
best_dice = 0.0
no_improve = 0
start_epoch = 1
history = {k: [] for k in ['train_loss', 'val_loss', 'dice', 'iou',
                            'precision', 'recall', 'specificity']}

if os.path.exists(LAST_MODEL_PATH):
    ckpt = torch.load(LAST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_dice = ckpt['best_dice']
    history = ckpt['history']
    no_improve = ckpt.get('no_improve', 0)
    print(f"Resumed from epoch {start_epoch - 1} | best Dice: {best_dice:.4f}")

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
train_start = time.time()

for epoch in range(start_epoch, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = combined_loss(model(images), masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    accum = MetricAccumulator()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            val_loss += combined_loss(outputs, masks).item()
            accum.update(outputs, masks)
    
    avg_val_loss = val_loss / len(val_loader)
    metrics = accum.summary()
    
    history['val_loss'].append(avg_val_loss)
    for k, v in metrics.items():
        history[k].append(v)
    
    enc_lr = optimizer.param_groups[0]['lr']
    dec_lr = optimizer.param_groups[1]['lr']
    print(f"\nEpoch {epoch} | Enc LR: {enc_lr:.2e} | Dec LR: {dec_lr:.2e}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"  Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_dice': best_dice,
        'history': history,
        'no_improve': no_improve,
    }, LAST_MODEL_PATH)
    
    # Save best model
    if metrics['dice'] > best_dice:
        best_dice = metrics['dice']
        no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("  ✅ Best model saved!")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

train_end = time.time()

# ─────────────────────────────────────────────
# Save training metrics
# ─────────────────────────────────────────────
pd.DataFrame(history).to_csv(METRICS_CSV, index_label='epoch')

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0, 0].plot(history['train_loss'], label='Train')
axes[0, 0].plot(history['val_loss'], label='Val')
axes[0, 0].set_title('Loss')
axes[0, 0].legend()
axes[0, 1].plot(history['dice'], label='Dice')
axes[0, 1].plot(history['iou'], label='IoU')
axes[0, 1].set_title('Dice & IoU')
axes[0, 1].legend()
axes[1, 0].plot(history['precision'], label='Precision')
axes[1, 0].plot(history['recall'], label='Recall')
axes[1, 0].set_title('Precision & Recall')
axes[1, 0].legend()
axes[1, 1].plot(history['specificity'], label='Specificity')
axes[1, 1].set_title('Specificity')
axes[1, 1].legend()
plt.tight_layout()
plt.savefig(f'training_metrics_{ENCODER_NAME}.png')
plt.show()

print(f"\n✅ Training complete!")
print(f"Training time: {train_end - train_start:.2f}s")
print(f"Best validation Dice: {best_dice:.4f}")
