from utils.data_loader import get_dataloaders
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    # Load datasets
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=2)

    # Check dataset sizes
    print(f"✅ Train Dataset Size: {len(train_loader.dataset)}")
    print(f"✅ Validation Dataset Size: {len(val_loader.dataset)}")
    print(f"✅ Test Dataset Size: {len(test_loader.dataset)}")

    # Get a batch from train_loader
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Check if the batch was loaded correctly
    print(f"✅ Loaded {len(images)} images from train_loader")
    print(f"✅ Labels: {labels}")

    # Plot the first image
    plt.imshow(images[0].permute(1, 2, 0))  # Convert tensor to image format
    plt.title(f"Class Label: {labels[0].item()}")
    plt.axis("off")
    plt.show()
