import torch

def extract_features(model, dataloader):
    model.eval()
    features = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.cuda()
            encoded_features = model.encoder(images)
            features.append(encoded_features.cpu().numpy())

    return features
