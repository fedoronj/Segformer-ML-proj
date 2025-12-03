import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

# -----------------------------
# 1. IMPORT YOUR CUSTOM MODEL
# -----------------------------
# Edit this line depending on your file names
from modeling_pathformer import PathFormer  


# -----------------------------
# 2. Dataset transforms
# -----------------------------
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.PILToTensor(),
])


# -----------------------------
# 3. Collate function
# -----------------------------
def collate_fn(batch):
    images = []
    lane_masks = []

    for sample in batch:
        img = image_transform(sample["image"])
        mask = mask_transform(sample["lane"])  # dataset key
        mask = (mask > 0).long()               # convert to {0,1}

        images.append(img)
        lane_masks.append(mask.squeeze(0))     # remove channel dim

    return torch.stack(images), torch.stack(lane_masks)


# -----------------------------
# 4. Load dataset
# -----------------------------
def load_data(batch_size=4):
    ds = load_dataset("bnsapa/road-detection")
    train_ds = ds["train"]
    val_ds   = ds["validation"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader


# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
def train(model, train_loader, val_loader, epochs=10, lr=3e-4, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)               # model outputs lane logits
            logits = logits.squeeze(1)           # [B,1,H,W] → [B,H,W]

            loss = criterion(logits, masks.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch} | Training Loss: {running_loss / len(train_loader):.4f}")

        # save checkpoint each epoch
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")


# -----------------------------
# 6. MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    train_loader, val_loader = load_data(batch_size=4)

    print("Loading model...")
    model = PathFormer()   # your custom SegFormer+SCNN class

    print("Starting training...")
    train(model, train_loader, val_loader)
