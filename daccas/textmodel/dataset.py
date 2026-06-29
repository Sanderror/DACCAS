"""
Dataset for text CAPTCHA images.
Supports two modes:
  1. Directory mode: images in a folder, labels extracted from filenames
  2. Manifest mode: TSV file with <filepath>\\t<label> per line (preferred)
"""
import os
import re
import random
import math
import numbers

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================================
# Data Augmentation (adapted from ABINet's transforms.py)
# ============================================================================
def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude

def sample_sym(magnitude, size=None):
    return (np.random.beta(4, 4, size=size) - 0.5) * 2 * magnitude

def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)

def get_interpolation(type='random'):
    if type == 'random':
        choice = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        return choice[random.randint(0, len(choice) - 1)]
    return cv2.INTER_LINEAR


class CVRandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, img):
        angle = sample_sym(self.degrees)
        src_h, src_w = img.shape[:2]
        M = cv2.getRotationMatrix2D(center=(src_w / 2, src_h / 2), angle=angle, scale=1.0)
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        dst_w = int(src_h * abs_sin + src_w * abs_cos)
        dst_h = int(src_h * abs_cos + src_w * abs_sin)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2
        flags = get_interpolation()
        return cv2.warpAffine(img, M, (dst_w, dst_h), flags=flags,
                              borderMode=cv2.BORDER_REPLICATE)


class CVRandomAffine:
    def __init__(self, degrees=15, translate=(0.3, 0.3), scale=(0.5, 2.), shear=(45, 15)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear if isinstance(shear, (list, tuple)) else [shear]

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        angle = sample_sym(self.degrees)
        scale = sample_uniform(self.scale[0], self.scale[1]) if self.scale else 1.0
        
        if len(self.shear) == 1:
            shear = [sample_sym(self.shear[0]), 0.]
        else:
            shear = [sample_sym(self.shear[0]), sample_sym(self.shear[1])]

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]
        cx, cy = src_w / 2, src_h / 2

        from numpy import sin, cos, tan
        a = cos(rot - sy) / cos(sy)
        b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
        c = sin(rot - sy) / cos(sy)
        d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

        M = np.array([d, -b, 0, -c, a, 0], dtype=np.float64).reshape(2, 3)
        M = M / scale

        M[0, 2] += cx - (M[0, 0] * cx + M[0, 1] * cy)
        M[1, 2] += cy - (M[1, 0] * cx + M[1, 1] * cy)

        flags = get_interpolation()
        return cv2.warpAffine(img, M, (src_w, src_h), flags=flags,
                              borderMode=cv2.BORDER_REPLICATE)


class CVRandomPerspective:
    def __init__(self, distortion=0.5):
        self.distortion = distortion

    def __call__(self, img):
        height, width = img.shape[:2]
        offset_h = sample_asym(self.distortion * height / 2, size=4).astype(np.int32)
        offset_w = sample_asym(self.distortion * width / 2, size=4).astype(np.int32)
        topleft = (offset_w[0], offset_h[0])
        topright = (width - 1 - offset_w[1], offset_h[1])
        botright = (width - 1 - offset_w[2], height - 1 - offset_h[2])
        botleft = (offset_w[3], height - 1 - offset_h[3])
        startpoints = np.array([(0, 0), (width - 1, 0),
                                (width - 1, height - 1), (0, height - 1)], dtype=np.float32)
        endpoints = np.array([topleft, topright, botright, botleft], dtype=np.float32)
        M = cv2.getPerspectiveTransform(startpoints, endpoints)
        rect = cv2.minAreaRect(endpoints)
        bbox = cv2.boxPoints(rect).astype(np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = max(bbox[:, 0].min(), 0), max(bbox[:, 1].min(), 0)
        flags = get_interpolation()
        img = cv2.warpPerspective(img, M, (max_x, max_y), flags=flags,
                                  borderMode=cv2.BORDER_REPLICATE)
        img = img[min_y:, min_x:]
        return img


class CVGeometry:
    def __init__(self, degrees=15, translate=(0.3, 0.3), scale=(0.5, 2.),
                 shear=(45, 15), distortion=0.5, p=0.5):
        self.p = p
        type_p = random.random()
        if type_p < 0.33:
            self.transform = CVRandomRotation(degrees=degrees)
        elif type_p < 0.66:
            self.transform = CVRandomAffine(degrees=degrees, translate=translate,
                                            scale=scale, shear=shear)
        else:
            self.transform = CVRandomPerspective(distortion=distortion)

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            return Image.fromarray(self.transform(img))
        return img


class CVGaussianNoise:
    def __init__(self, mean=0, var=20):
        self.mean = mean
        self.var = max(int(sample_asym(var)), 1)

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.var ** 0.5, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)


class CVMotionBlur:
    def __init__(self, degrees=12, angle=90):
        self.degree = max(int(sample_asym(degrees)), 1)
        self.angle = sample_uniform(-angle, angle)

    def __call__(self, img):
        M = cv2.getRotationMatrix2D((self.degree // 2, self.degree // 2), self.angle, 1)
        kernel = np.zeros((self.degree, self.degree))
        kernel[self.degree // 2, :] = 1
        kernel = cv2.warpAffine(kernel, M, (self.degree, self.degree))
        kernel = kernel / self.degree
        img = cv2.filter2D(img, -1, kernel)
        return np.clip(img, 0, 255).astype(np.uint8)


class CVDeterioration:
    def __init__(self, var=20, degrees=12, factor=4, p=0.5):
        self.p = p
        t = []
        if var is not None:
            t.append(CVGaussianNoise(var=var))
        if degrees is not None:
            t.append(CVMotionBlur(degrees=degrees))
        random.shuffle(t)
        self.transforms = t

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            for t in self.transforms:
                img = t(img)
            return Image.fromarray(img)
        return img


class CVColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5):
        self.p = p
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img


# ============================================================================
# Character Mapping
# ============================================================================
class CharsetMapper:
    """Maps characters to integer labels and back.
    
    Label 0 is reserved as the null/padding/EOS token.
    Character labels start from 1.
    """
    def __init__(self, charset_path, max_length=26, null_char=u'\u2591'):
        self.null_char = null_char
        self.max_length = max_length
        self.null_label = 0

        self.label_to_char = {0: null_char}
        self.char_to_label = {null_char: 0}

        pattern = re.compile(r'(\d+)\t(.+)')
        with open(charset_path, 'r') as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    label = int(m.group(1)) + 1
                    char = m.group(2)
                    self.label_to_char[label] = char
                    self.char_to_label[char] = label

        self.num_classes = len(self.label_to_char)

    def get_labels(self, text, padding=True, lowercase=False):
        """Convert text to label sequence, with padding to max_length."""
        if lowercase:
            text = text.lower()
        labels = []
        for ch in text:
            if ch in self.char_to_label:
                labels.append(self.char_to_label[ch])
            # Skip unknown characters
        if padding:
            labels = labels + [self.null_label] * (self.max_length - len(labels))
        return labels[:self.max_length]

    def get_text(self, labels, trim=True):
        """Convert label sequence back to text string."""
        chars = []
        for l in labels:
            l = l.item() if isinstance(l, torch.Tensor) else int(l)
            if l == self.null_label:
                break
            if l in self.label_to_char:
                chars.append(self.label_to_char[l])
        return ''.join(chars)

    def get_length(self, labels):
        """Get the effective length (position of first null)."""
        for i, l in enumerate(labels):
            l_val = l.item() if isinstance(l, torch.Tensor) else int(l)
            if l_val == self.null_label:
                return i
        return len(labels)


# ============================================================================
# Dataset
# ============================================================================
class CaptchaDataset(Dataset):
    """Dataset for CAPTCHA images.
    
    Supports two modes:
      1. manifest_path: a TSV file with lines "<filepath>\\t<label>"
         Each filepath is either absolute or relative to manifest's directory.
      2. image_dir + image_paths: directory scan (legacy mode)
    
    Args:
        manifest_path: path to TSV manifest file (preferred)
        image_dir: directory containing images (legacy mode)
        image_paths: list of image filenames (legacy mode)
        charset: CharsetMapper instance
        img_h: target image height (default 32)
        img_w: target image width (default 128)
        augment: whether to apply data augmentation
        lowercase: whether to lowercase labels (default False)
    """
    def __init__(self, manifest_path=None, image_dir=None, image_paths=None,
                 charset=None, img_h=32, img_w=128, augment=False, lowercase=False):
        self.charset = charset
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.lowercase = lowercase

        # Load data entries as list of (filepath, label)
        self.entries = []

        if manifest_path is not None:
            # Manifest mode: read TSV file
            with open(manifest_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        filepath, label = parts
                        self.entries.append((filepath, label))
        elif image_dir is not None:
            # Legacy directory mode
            if image_paths is None:
                exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
                image_paths = sorted([
                    f for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in exts
                ])
            for fname in image_paths:
                filepath = os.path.join(image_dir, fname)
                label = self._extract_label_from_filename(fname)
                self.entries.append((filepath, label))
        else:
            raise ValueError("Must provide either manifest_path or image_dir")

        # Build augmentation transforms
        self.color_jitter = CVColorJitter(p=0.5) if augment else None

        # Standard normalization
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def _extract_label_from_filename(filename):
        """Extract label from filename like 'AbC123_42.png' -> 'AbC123'.
        Assumes <label>_<ID>.ext format (label before last underscore).
        """
        name = os.path.splitext(filename)[0]
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0]
        return name

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filepath, label_str = self.entries[idx]

        # Load image
        img = Image.open(filepath).convert('RGB')

        # Apply augmentation
        if self.augment:
            geo = CVGeometry(degrees=15, translate=(0.3, 0.3),
                           scale=(0.5, 2.), shear=(45, 15),
                           distortion=0.5, p=0.5)
            img = geo(img)
            det = CVDeterioration(var=20, degrees=12, factor=None, p=0.5)
            img = det(img)
            if self.color_jitter:
                img = self.color_jitter(img)

        # Resize to target dimensions
        img = img.resize((self.img_w, self.img_h), Image.BILINEAR)

        # Convert to tensor and normalize
        img = self.to_tensor(img)
        img = self.normalize(img)

        # Encode label
        label = torch.tensor(
            self.charset.get_labels(label_str, lowercase=self.lowercase),
            dtype=torch.long
        )
        length = min(len(label_str), self.charset.max_length)

        return img, label, length
