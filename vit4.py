import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.models.vision_transformer import VisionTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torchvision.utils import make_grid
from timm.models.vision_transformer import VisionTransformer

class ViTWithAttention(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_maps = []

    def forward_features(self, x):
        self.attn_maps.clear()
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x, attn = blk.forward_with_attention(x)  # 👈 Unpack
            self.attn_maps.append(attn)

        x = self.norm(x)
        return x[:, 0, :]  # ✅ CLS token

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)  # Explicitly call classification head

def add_attention_forward_hook():
    from timm.models.vision_transformer import Block

    def forward_with_attention(self, x):
        B, N, C = x.shape
        qkv = self.attn.qkv(self.norm1(x)).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # q, k, v shape: [B, heads, tokens, dim]
        attn_weights = (q @ k.transpose(-2, -1)) * self.attn.scale
        attn_probs = attn_weights.softmax(dim=-1)  # [B, heads, tokens, tokens]

        x_attn = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.drop_path1(self.ls1(self.attn.proj(x_attn)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, attn_probs

    Block.forward_with_attention = forward_with_attention


add_attention_forward_hook()



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
        self.test_loader = DataLoader(self.test_data, batch_size=64, shuffle=False)

        self.model = ViTWithAttention(
            img_size=32,
            patch_size=16,
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

    def get_patch_embeddings(self, images):
        """Get output of the transformer before classification head."""
        self.model.eval()
        with torch.no_grad():
            return self.model.forward_features(images.to(self.device))  # Shape: [B, num_patches+1, embed_dim]
        
    def visualize_tsne(self, num_batches=5, show_images=True):
        """Run t-SNE on CLS token features and plot, with image thumbnails."""
        print("🔍 Extracting CLS token features for t-SNE...")
        self.model.eval()
        features = []
        labels = []
        raw_images = []

        with torch.no_grad():
            count = 0
            for images, lbls in self.test_loader:
                if count >= num_batches:
                    break
                out = self.get_patch_embeddings(images)  # [B, N+1, D]
                cls_tokens = out[:, 0, :]  # CLS token
                features.append(cls_tokens.cpu())
                labels.append(lbls.cpu())
                raw_images.append(images.cpu())
                count += 1

        features = torch.cat(features, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        raw_images = torch.cat(raw_images, dim=0)  # [N, 3, 32, 32]

        print("🔄 Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced = tsne.fit_transform(features)

        print("📊 Plotting...")
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=15)
        plt.legend(*scatter.legend_elements(), title="Digits")
        plt.title("t-SNE of CLS Features with Image Thumbnails")

        if show_images:
            shown_imgs = np.array([[1., 1.]])  # used to avoid overlapping images
            for i in range(len(reduced)):
                dist = np.sum((reduced[i] - shown_imgs) ** 2, 1)
                if np.min(dist) < 0.01:  # avoid cluttering
                    continue
                shown_imgs = np.r_[shown_imgs, [reduced[i]]]

                img = raw_images[i].permute(1, 2, 0).numpy()
                img = img.squeeze() if img.shape[2] == 1 else img  # handle grayscale
                imagebox = OffsetImage(img, zoom=0.8, cmap='gray')
                ab = AnnotationBbox(imagebox, reduced[i], frameon=False)
                ax.add_artist(ab)

        plt.show()

    def visualize_attention_map(self, image_idx=0, layer_idx=0):
        import matplotlib.pyplot as plt

        self.model.eval()
        images, labels = next(iter(self.test_loader))
        image = images[image_idx:image_idx+1].to(self.device)

        # Only extract attention features, not logits
        with torch.no_grad():
            _ = self.model.forward_features(image)

        if not self.model.attn_maps:
            print("⚠️ Attention maps not recorded.")
            return

        attn = self.model.attn_maps[layer_idx]  # Expect shape: [B, heads, tokens, tokens]
        print("🔍 Attn shape:", attn.shape)

        # Average over heads → shape: [tokens, tokens]
        attn_mean = attn[0].mean(0)  # shape: [tokens, tokens]
        cls_to_patch = attn_mean[0, 1:]  # CLS token attends to all patches

        num_patches = cls_to_patch.shape[0]
        grid_size = int(num_patches ** 0.5)
        heatmap = cls_to_patch.reshape(grid_size, grid_size).cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(images[image_idx].permute(1, 2, 0).squeeze(), cmap='gray')
        ax[0].set_title(f"Digit: {labels[image_idx].item()}")
        ax[1].imshow(heatmap, cmap='inferno')
        ax[1].set_title(f"CLS Attention (Layer {layer_idx})")
        plt.tight_layout()
        plt.show()

    def visualize_patch_clusters(self, num_images=200, num_clusters=10):
        self.model.eval()
        patch_embeddings = []
        patch_images = []

        print(f"🔍 Collecting patches from {num_images} test images...")
        count = 0
        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.to(self.device)
                B = images.size(0)
                feats = self.model.forward_features(images)  # [B, D] (CLS token only)

                # Recompute full token features to get patch embeddings
                x = self.model.patch_embed(images)  # [B, num_patches, D]
                x = x + self.model.pos_embed[:, 1:, :]  # exclude CLS token from pos_embed

                x = x.cpu().numpy()  # [B, num_patches, D]
                for i in range(B):
                    for p in range(x.shape[1]):
                        patch = images[i, :, :, :].cpu().numpy().transpose(1, 2, 0)  # HWC
                        patch_gray = patch.mean(axis=2) if patch.shape[2] == 3 else patch.squeeze()
                        
                        y, x_ = divmod(p, 2)
                        # patch_crop = patch[y*4:(y+1)*4, x_*4:(x_+1)*4, :]
                        patch_crop = patch[y*16:(y+1)*16, x_*8:(x_+1)*16, :]  # ✅ update from 4 to 8


                        # Now compute variance on the patch crop itself, not full image
                        patch_crop_gray = patch_crop.mean(axis=2) if patch_crop.shape[2] == 3 else patch_crop.squeeze()
                        if patch_crop_gray.var() < 0.01:
                            continue  # Skip flat patches

                        patch_embeddings.append(x[i, p])  # ✅ Only save embedding if patch passes
                        patch_images.append(patch_crop)  # Save the patch image

                count += B
                if count >= num_images:
                    break

        patch_embeddings = np.array(patch_embeddings)
        print(f"✅ Collected {len(patch_embeddings)} patches")

        print("🔄 Reducing patch embeddings via t-SNE...")
        reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(patch_embeddings)

        print("🔗 Clustering patches with KMeans...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(reduced)

        print("🖼️ Visualizing clusters...")
        fig, axs = plt.subplots(num_clusters, 10, figsize=(15, num_clusters * 1.5))

        for cluster_id in range(num_clusters):
            indices = np.where(labels == cluster_id)[0]
            selected = np.random.choice(indices, min(10, len(indices)), replace=False)

            for j, idx in enumerate(selected):
                patch = patch_images[idx]
                axs[cluster_id, j].imshow(patch.squeeze(), cmap='gray')
                axs[cluster_id, j].axis('off')

            for j in range(len(selected), 10):
                axs[cluster_id, j].axis('off')

            axs[cluster_id, 0].set_ylabel(f"Cluster {cluster_id}", fontsize=10)

        plt.tight_layout()
        plt.suptitle("Patch-Level Cluster Visualization (t-SNE + KMeans)", fontsize=14)
        plt.subplots_adjust(top=0.95)
        plt.show()
        
if __name__ == "__main__":
    classifier = ViTMNISTClassifier()
    # classifier.train(epochs=5)
    # classifier.save_model()  # Save this new one for future 8×8 patch use

    classifier.load_model()
    classifier.evaluate()
    # classifier.visualize_attention_map(image_idx=3)
    classifier.visualize_patch_clusters(num_images=200, num_clusters=10)
    # classifier.visualize_tsne(num_batches=5)

