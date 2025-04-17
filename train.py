import torch
from utils.data_loader import get_dataloaders
from models.context_encoder import train_context_encoder
from models.feature_extractor import extract_features
from models.pymft_classifier import classify_with_pymft

# ✅ Force CPU Training
device = torch.device("cpu")  
print(f"🚀 Using device: {device}")

# ✅ Load dataset with CPU-friendly settings
train_loader, val_loader, _ = get_dataloaders(batch_size=8)  # Reduce batch size

# ✅ Train context encoder
context_encoder = train_context_encoder(train_loader)
context_encoder.to(device)

# ✅ Extract features from segmented images
train_features = extract_features(context_encoder, train_loader)

# ✅ Get labels correctly
train_labels = [label for _, label in train_loader]  # Fix label extraction

# ✅ Classify using PyMFT
classify_with_pymft(train_features, train_labels)

print("🎉 Training completed successfully!")
