import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class Config:
    IMG_SIZE = 224
    CLASSES = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']
    CLASS_NAMES = {
        'cbb': 'Cassava Bacterial Blight',
        'cbsd': 'Cassava Brown Streak Disease',
        'cgm': 'Cassava Green Mite',
        'cmd': 'Cassava Mosaic Disease',
        'healthy': 'Healthy'
    }


def build_feature_extractor(device):
    weights = VGG16_Weights.IMAGENET1K_V1
    backbone = vgg16(weights=weights)
    model = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten()).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_model(preferred_path, fallback_path):
    if os.path.exists(preferred_path):
        with open(preferred_path, 'rb') as f:
            return pickle.load(f)
    with open(fallback_path, 'rb') as f:
        return pickle.load(f)


class CassavaPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = Config()
        self.extractor = build_feature_extractor(self.device)

        with open('outputs/feature_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open('outputs/pca_model.pkl', 'rb') as f:
            self.pca = pickle.load(f)

        self.model = load_model('outputs/random_forest_optimized.pkl', 'outputs/random_forest_model.pkl')

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)

    def _extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.config.IMG_SIZE, self.config.IMG_SIZE)).astype(np.float32)
        batch = np.expand_dims(img, axis=0) / 255.0
        batch = (batch - self.mean) / self.std
        batch = np.transpose(batch, (0, 3, 1, 2))

        tensor = torch.from_numpy(batch)
        with torch.inference_mode():
            features = self.extractor(tensor.to(self.device)).cpu().numpy()

        features = self.scaler.transform(features)
        features = self.pca.transform(features)
        return features

    def predict(self, image_path):
        features = self._extract(image_path)
        pred = int(self.model.predict(features)[0])
        proba = self.model.predict_proba(features)[0]
        label = self.config.CLASSES[pred]
        return {
            'class': label,
            'class_full': self.config.CLASS_NAMES[label],
            'confidence': float(proba[pred]),
            'probabilities': {self.config.CLASSES[i]: float(proba[i]) for i in range(len(self.config.CLASSES))}
        }


def main():
    parser = argparse.ArgumentParser(description='Cassava leaf disease prediction')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('-o', '--output', type=str, default='', help='Optional JSON output path')
    args = parser.parse_args()

    predictor = CassavaPredictor()
    result = predictor.predict(args.image_path)
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
