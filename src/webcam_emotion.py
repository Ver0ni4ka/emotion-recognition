from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


IMAGE_SIZE = 224
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_resnet18.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path, device: torch.device):
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


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_face(face_bgr, model, classes, transform, device):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_rgb)

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    predicted_class = classes[predicted_idx]
    return predicted_class, confidence


def main():
    print("Устройство:", device)

    if not MODEL_PATH.exists():
        print(f"Модель не найдена: {MODEL_PATH}")
        return

    model, classes = load_model(MODEL_PATH, device)
    transform = get_transform()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    print("Камера запущена. Нажми 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]

            if face.size == 0:
                continue

            predicted_class, confidence = predict_face(
                face, model, classes, transform, device
            )

            label = f"{predicted_class}: {confidence:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10 if y - 10 > 10 else y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2
            )

        cv2.imshow("Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()