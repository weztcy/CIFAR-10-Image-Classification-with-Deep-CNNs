"""
Prediction and inference module for single or batch images.
Loads model checkpoint, applies preprocessing, and returns top predictions.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

try:
    from src.models import BaselineCNN, ResNetCIFAR
except ImportError:
    from models import BaselineCNN, ResNetCIFAR


class CIFARPredictor:
    """Predictor class for CIFAR-10 classification."""
    def __init__(self, checkpoint_path=None, model_type='resnet'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(12)

        if checkpoint_path is None:
            checkpoint_path = f"outputs/checkpoints/{model_type}_best.pt"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']

        if model_type == 'baseline':
            self.model = BaselineCNN(num_classes=len(self.class_names))
        else:
            self.model = ResNetCIFAR(num_classes=len(self.class_names))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def predict(self, image_input):
        """
        image_input: PIL Image or filepath str or np.ndarray (H, W, 3)
        Returns:
            top_class: str
            top_prob: float
            all_probs: dict {class_name: prob}
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise TypeError("Unsupported image input type.")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        top_prob = float(probs[top_idx])

        all_probs = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        return top_class, top_prob, all_probs


if __name__ == '__main__':
    # Test predictor on a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    print("Testing CIFARPredictor...")
    # Check if checkpoint exists
    if os.path.exists("outputs/checkpoints/resnet_best.pt"):
        predictor = CIFARPredictor(model_type='resnet')
        top_c, top_p, probs = predictor.predict(dummy_img)
        print(f"Prediction: {top_c} ({top_p*100:.2f}%)")
    else:
        print("Checkpoint not found yet. Train model first.")
