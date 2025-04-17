import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Set your dataset directory path
DATASET_DIR = r"M:\4TH_Year Project\OCT2017"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

class OCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        # Allowed image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue  # Skip if it's not a folder
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path) and os.path.splitext(img_name)[1].lower() in image_extensions:
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert to PyTorch Tensor
])

# Create data loaders
def get_dataloaders(batch_size=32):
    train_dataset = OCTDataset(TRAIN_DIR, transform=transform)
    val_dataset = OCTDataset(VAL_DIR, transform=transform)
    test_dataset = OCTDataset(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
