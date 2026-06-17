from flask import Flask, render_template, request, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files.get("image")

        if file:

            filepath = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(filepath)

            image_path = "/uploads/" + file.filename

            image = Image.open(filepath).convert("RGB")

            img = transform(image)
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(img)

                probs = torch.softmax(
                    output,
                    dim=1
                )

                pred = torch.argmax(
                    probs,
                    dim=1
                ).item()

                confidence = round(
                    probs[0][pred].item() * 100,
                    2
                )

            if pred == 0:
                result = "FAKE"
            else:
                result = "REAL"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)