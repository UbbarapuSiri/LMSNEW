# ─────────────────────────────────────────────
# TESTING CELL - Run after training completes
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os
import json
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_NAME = "tf_efficientnet_b7"  # Corrected name
MODEL_PATH = f"best_model_{ENCODER_NAME}.pth"
IMG_SIZE = 384
BATCH_SIZE = 8

# Mean/Std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Paths
BASE_PATH = '/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/segmentation_task'
TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test', 'images')
TEST_MASK_DIR = os.path.join(BASE_PATH, 'test', 'masks')
CROP_CACHE = "ecc_crop_cache.json"

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def get_base_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def match_pairs(image_files, mask_files):
    mask_dict = {get_base_name(f): f for f in mask_files}
    pairs = [(img, mask_dict[get_base_name(img)]) for img in image_files if get_base_name(img) in mask_dict]
    imgs, masks = zip(*pairs)
    return list(imgs), list(masks)

def list_images(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def list_masks(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) 
                   if f.lower().endswith('.png')])

def get_tumor_type(image_path):
    """Extract tumor type from image filename."""
    filename = os.path.basename(image_path)
    if '_gl_' in filename:
        return 'Glioma'
    elif '_me_' in filename:
        return 'Meningioma'
    elif '_pi_' in filename:
        return 'Pituitary'
    else:
        return 'Unknown'

# ─────────────────────────────────────────────
# Test Dataset Class
# ─────────────────────────────────────────────
class BRISCTestDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, crop_cache=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.crop_cache = crop_cache or {}
    
    def _apply_crop(self, image, mask, mask_path):
        bbox = self.crop_cache.get(mask_path)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        return image, mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        image, mask = self._apply_crop(image, mask, self.mask_paths[idx])
        image = np.stack([image, image, image], axis=-1)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        return image, mask.unsqueeze(0), self.image_paths[idx]

# ─────────────────────────────────────────────
# Model Definition (same as training)
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
    def __init__(self, encoder_name="tf_efficientnet_b7", pretrained=False,
                 decoder_channels=(512, 256, 128, 64), num_classes=1):
        super().__init__()
        
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feats = self.encoder(dummy)
            enc_channels = [f.shape[1] for f in feats]
        
        self.decoder4 = DecoderBlock(enc_channels[3], enc_channels[2], decoder_channels[0])
        self.decoder3 = DecoderBlock(decoder_channels[0], enc_channels[1], decoder_channels[1])
        self.decoder2 = DecoderBlock(decoder_channels[1], enc_channels[0], decoder_channels[2])
        self.decoder1 = DecoderBlock(decoder_channels[2], 0, decoder_channels[3])
        self.head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
    
    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        d4 = self.decoder4(f4, f3)
        d3 = self.decoder3(d4, f2)
        d2 = self.decoder2(d3, f1)
        d1 = self.decoder1(d2)
        out = F.interpolate(self.head(d1), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

# ─────────────────────────────────────────────
# Metric Accumulator
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
# TTA Function
# ─────────────────────────────────────────────
def tta_predict(model, images):
    """Average predictions over 4 augmentations."""
    preds = []
    for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
        x = images.clone()
        if flip_h:
            x = torch.flip(x, dims=[3])
        if flip_v:
            x = torch.flip(x, dims=[2])
        p = torch.sigmoid(model(x))
        if flip_h:
            p = torch.flip(p, dims=[3])
        if flip_v:
            p = torch.flip(p, dims=[2])
        preds.append(p)
    return torch.stack(preds).mean(dim=0)

# ─────────────────────────────────────────────
# Main Testing Function
# ─────────────────────────────────────────────
def test_model():
    print("="*70)
    print(f"Testing {ENCODER_NAME.upper()} + UNet++ on BRISC Dataset")
    print("="*70)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    test_image_files = list_images(TEST_IMAGE_DIR)
    test_mask_files = list_masks(TEST_MASK_DIR)
    test_image_files, test_mask_files = match_pairs(test_image_files, test_mask_files)
    print(f"Found {len(test_image_files)} test pairs")
    
    # Load crop cache
    with open(CROP_CACHE, 'r') as f:
        crop_cache = json.load(f)
    
    # Create test dataset
    test_transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    test_dataset = BRISCTestDataset(
        test_image_files, test_mask_files, 
        transform=test_transform, crop_cache=crop_cache
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Load model
    print("\n[2/4] Loading model...")
    model = EfficientNetUNetPlusPlus(
        encoder_name=ENCODER_NAME,
        pretrained=False,
        decoder_channels=(512, 256, 128, 64),
        num_classes=1,
    ).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ Model loaded from: {MODEL_PATH}")
    
    # Group by tumor type
    print("\n[3/4] Grouping images by tumor type...")
    tumor_accumulators = {
        'Glioma': MetricAccumulator(),
        'Meningioma': MetricAccumulator(),
        'Pituitary': MetricAccumulator()
    }
    overall_accum = MetricAccumulator()
    per_image_metrics = []
    
    # Run inference
    print("\n[4/4] Running inference with TTA...")
    test_start = time.time()
    
    with torch.no_grad():
        for images, masks, paths in tqdm(test_loader, desc="Testing with TTA"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # TTA prediction
            avg_probs = tta_predict(model, images)
            
            # Convert to logits for metric accumulation
            logit_preds = torch.log(
                avg_probs.clamp(1e-6, 1 - 1e-6) / (1 - avg_probs.clamp(1e-6, 1 - 1e-6))
            )
            
            # Update overall metrics
            overall_accum.update(logit_preds, masks)
            
            # Calculate per-image metrics
            for i in range(images.size(0)):
                tumor_type = get_tumor_type(paths[i])
                if tumor_type == 'Unknown':
                    continue
                
                pred_single = logit_preds[i:i+1]
                target_single = masks[i:i+1]
                pred_binary = (torch.sigmoid(pred_single) > 0.5).float()
                target_binary = target_single.float()
                
                tp = (pred_binary * target_binary).sum().item()
                fp = (pred_binary * (1 - target_binary)).sum().item()
                fn = ((1 - pred_binary) * target_binary).sum().item()
                tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
                
                dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
                iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
                precision = (tp + 1e-7) / (tp + fp + 1e-7)
                recall = (tp + 1e-7) / (tp + fn + 1e-7)
                specificity = (tn + 1e-7) / (tn + fp + 1e-7)
                
                per_image_metrics.append({
                    'image_id': os.path.basename(paths[i]),
                    'tumor_type': tumor_type,
                    'dice': dice,
                    'iou': iou,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity
                })
                
                if tumor_type in tumor_accumulators:
                    tumor_accumulators[tumor_type].tp += tp
                    tumor_accumulators[tumor_type].fp += fp
                    tumor_accumulators[tumor_type].fn += fn
                    tumor_accumulators[tumor_type].tn += tn
    
    test_end = time.time()
    
    # Calculate final metrics
    overall_metrics = overall_accum.summary()
    tumor_metrics = {ttype: accum.summary() for ttype, accum in tumor_accumulators.items()}
    
    # Create DataFrame
    df_per_image = pd.DataFrame(per_image_metrics)
    tumor_stats = df_per_image.groupby('tumor_type').agg({
        'dice': ['mean', 'std', 'count'],
        'iou': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'specificity': ['mean', 'std']
    }).round(4)
    
    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    print("\n📊 OVERALL PERFORMANCE (with TTA):")
    print("-" * 50)
    for metric, value in overall_metrics.items():
        print(f"  {metric.capitalize():<12}: {value:.4f} ({value*100:.2f}%)")
    
    print("\n📈 PER-TUMOR TYPE PERFORMANCE:")
    print("-" * 85)
    print(f"{'Tumor Type':<12} {'Dice (%)':<12} {'IoU (%)':<12} {'Precision (%)':<12} {'Recall (%)':<12} {'Specificity (%)':<12}")
    print("-" * 85)
    
    for ttype in ['Glioma', 'Meningioma', 'Pituitary']:
        if ttype in tumor_metrics:
            m = tumor_metrics[ttype]
            print(f"{ttype:<12} {m['dice']*100:>8.2f}%    {m['iou']*100:>8.2f}%    {m['precision']*100:>8.2f}%    {m['recall']*100:>8.2f}%    {m['specificity']*100:>8.2f}%")
    
    print("-" * 85)
    print(f"\n⏱️  Testing Time: {test_end - test_start:.2f} seconds")
    
    # Save results
    print("\n💾 Saving results...")
    
    with open(f'test_results_{ENCODER_NAME}.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"Test Results — {ENCODER_NAME.upper()} + UNet++ (with TTA)\n")
        f.write("="*70 + "\n\n")
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-"*50 + "\n")
        for metric, value in overall_metrics.items():
            f.write(f"  {metric.capitalize():<12}: {value:.4f} ({value*100:.2f}%)\n")
        f.write("\nPER-TUMOR TYPE PERFORMANCE:\n")
        f.write("-"*85 + "\n")
        f.write(f"{'Tumor Type':<12} {'Dice':<12} {'IoU':<12} {'Precision':<12} {'Recall':<12} {'Specificity':<12}\n")
        f.write("-"*85 + "\n")
        for ttype in ['Glioma', 'Meningioma', 'Pituitary']:
            if ttype in tumor_metrics:
                m = tumor_metrics[ttype]
                f.write(f"{ttype:<12} {m['dice']:.4f}     {m['iou']:.4f}     {m['precision']:.4f}     {m['recall']:.4f}     {m['specificity']:.4f}\n")
        f.write("-"*85 + "\n")
        f.write(f"\nTesting Time: {test_end - test_start:.2f} seconds\n")
    
    df_per_image.to_csv(f'per_image_metrics_{ENCODER_NAME}.csv', index=False)
    tumor_stats.to_csv(f'tumor_type_summary_{ENCODER_NAME}.csv')
    
    print(f"✅ Results saved to:")
    print(f"   - test_results_{ENCODER_NAME}.txt")
    print(f"   - per_image_metrics_{ENCODER_NAME}.csv")
    print(f"   - tumor_type_summary_{ENCODER_NAME}.csv")
    
    # LaTeX table
    print("\n" + "="*70)
    print("LaTeX TABLE FOR PAPER")
    print("="*70)
    print("\\begin{table}[h]")
    print("\\captionsetup{justification=centering, font=small}")
    print(f"\\caption{{Segmentation Performance of EfficientNet-B7 on BRISC Dataset}}\\label{{tab:efficientnet_b7_seg}}")
    print("\\centering")
    print("\\begin{tabular}{@{}lccccc@{}}")
    print("\\toprule")
    print("Metric & Dice (\\%) & IoU (\\%) & Precision (\\%) & Recall (\\%) & Specificity (\\%) \\\\")
    print("\\midrule")
    print(f"Overall & {overall_metrics['dice']*100:.2f} & {overall_metrics['iou']*100:.2f} & {overall_metrics['precision']*100:.2f} & {overall_metrics['recall']*100:.2f} & {overall_metrics['specificity']*100:.2f} \\\\")
    print("\\midrule")
    for ttype in ['Glioma', 'Meningioma', 'Pituitary']:
        if ttype in tumor_metrics:
            m = tumor_metrics[ttype]
            print(f"{ttype} & {m['dice']*100:.2f} & {m['iou']*100:.2f} & {m['precision']*100:.2f} & {m['recall']*100:.2f} & {m['specificity']*100:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)
    
    return overall_metrics, tumor_metrics, df_per_image

# Run test
if __name__ == "__main__":
    overall_metrics, tumor_metrics, df_per_image = test_model()
