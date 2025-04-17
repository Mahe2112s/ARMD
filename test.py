from utils.data_loader import get_dataloader
from models.feature_extractor import extract_features
from models.pymft_classifier import classify_with_pymft
import torch

test_loader = get_dataloader("data/test", batch_size=32)

# Load trained model
context_encoder = torch.load("models/context_encoder.pth").cuda()

# Extract features
test_features = extract_features(context_encoder, test_loader)

# Classification
test_labels = [label for _, label in test_loader.dataset]
classify_with_pymft(test_features, test_labels)
