import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.models.vision_transformer import VisionTransformer
import os


class ViTMNISTClassifier:
    def __init__(self, model_path="vit_mnist.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("CUDA available:", torch.cuda.is_available())
        print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

        self.model_path = model_path

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.train_loader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=64)

        self.model = VisionTransformer(
            img_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=10,
            embed_dim=192,
            depth=7,
            num_heads=3,
            mlp_ratio=2.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch + 1}] Loss: {total_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No saved model found at {self.model_path}")

if __name__ == "__main__":
    classifier = ViTMNISTClassifier()
    # classifier.train(epochs=5)
    # classifier.evaluate()
    # classifier.save_model()

    # print("\nReloading model and re-evaluating:")
    # new_classifier = ViTMNISTClassifier()
    classifier.load_model()
    classifier.evaluate()