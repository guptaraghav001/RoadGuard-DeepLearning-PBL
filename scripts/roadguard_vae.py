
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(
    "data/processed/cnn_crops_complete/train"
)
OUTPUT = Path("experiments/vae")
OUTPUT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64
LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 5
MAX_IMAGES = 2000

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class RoadDamageDataset(Dataset):
    def __init__(self):
        self.paths = sorted(
            DATA_DIR.glob("class_*/*.png")
        )

        random.shuffle(self.paths)
        self.paths = self.paths[:MAX_IMAGES]

        if not self.paths:
            raise RuntimeError(
                "No training images found"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            img = img.convert("RGB")
            img = img.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )
            array = np.asarray(
                img, dtype=np.float32
            ) / 255.0

        return torch.from_numpy(
            array.transpose(2, 0, 1).copy()
        )

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, LATENT_DIM)
        self.logvar = nn.Linear(256, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(
                256, 3 * IMAGE_SIZE * IMAGE_SIZE
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        hidden = self.encoder(x)

        mu = self.mu(hidden)
        logvar = self.logvar(hidden)

        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)

        z = mu + noise * std

        reconstruction = self.decoder(z)

        reconstruction = reconstruction.view(
            -1, 3, IMAGE_SIZE, IMAGE_SIZE
        )

        return reconstruction, mu, logvar

def loss_function(recon, x, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(
        recon, x, reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + kl_loss

def save_grid(images, path):
    images = (
        images.detach()
        .cpu()
        .clamp(0, 1)
        .numpy()
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE * 4, IMAGE_SIZE * 4)
    )

    for i in range(min(16, len(images))):
        array = (
            images[i]
            .transpose(1, 2, 0)
            * 255
        ).astype(np.uint8)

        tile = Image.fromarray(array)

        canvas.paste(
            tile,
            (
                (i % 4) * IMAGE_SIZE,
                (i // 4) * IMAGE_SIZE
            )
        )

    canvas.save(path)

def main():
    dataset = RoadDamageDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = VAE().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("===== ROADGUARD VAE TRAINING =====")
    print("Device:", DEVICE)
    print("Images:", len(dataset))

    losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            recon, mu, logvar = model(batch)

            loss = loss_function(
                recon, batch, mu, logvar
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {avg_loss:.4f}"
        )

    model.eval()

    with torch.no_grad():
        random_latent = torch.randn(
            16, LATENT_DIM,
            device=DEVICE
        )

        generated = model.decoder(
            random_latent
        ).view(
            -1, 3, IMAGE_SIZE, IMAGE_SIZE
        )

    save_grid(
        generated,
        OUTPUT / "generated_samples.png"
    )

    torch.save(
        model.state_dict(),
        OUTPUT / "roadguard_vae.pt"
    )

    report = {
        "model": "Variational Autoencoder",
        "training_images": len(dataset),
        "epochs": EPOCHS,
        "latent_dimension": LATENT_DIM,
        "average_losses": losses,
        "generated_samples": (
            "generated_samples.png"
        ),
        "limitations": (
            "Small demonstration model. "
            "Generated images are not validated "
            "as realistic road damage samples."
        )
    }

    with open(
        OUTPUT / "vae_results.json",
        "w"
    ) as f:
        json.dump(report, f, indent=4)

    print("\nVAE EXPERIMENT COMPLETED")
    print("Generated images saved successfully")

if __name__ == "__main__":
    main()
