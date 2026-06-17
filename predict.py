import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

img_path = input("Enter image path: ")

image = Image.open(img_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

if prediction == 0:
    print("FAKE")
else:
    print("REAL")