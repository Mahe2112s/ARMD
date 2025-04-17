import numpy as np
from pymft.mft import MFT # type: ignore

def classify_with_pymft(features, labels):
    features = np.array(features).reshape(len(labels), -1)
    labels = np.array(labels)

    classifier = MFT()
    classifier.fit(features, labels)
    predictions = classifier.predict(features)

    accuracy = np.mean(predictions == labels)
    print(f"Classification Accuracy: {accuracy:.2f}")
