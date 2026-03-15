from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


IMAGE_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_resnet18.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, classes


def predict_image(image_path, model, classes, device):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    predicted_class = classes[predicted_idx]
    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python src/infer.py путь_к_изображению")
        sys.exit(1)

    image_path = sys.argv[1]

    model, classes = load_model(MODEL_PATH, device)
    predicted_class, confidence = predict_image(image_path, model, classes, device)

    print("Устройство:", device)
    print("Предсказание:", predicted_class)
    print("Уверенность:", round(confidence, 4))