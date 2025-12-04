from torch.utils.data import DataLoader
import torch

dataset = load_dataset("tusimple", split="train", streaming=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train().cuda()

for epoch in range(10):
    for batch in dataloader:
        images = batch["pixel_values"].cuda()
        labels = batch["labels"].cuda()

        logits = model(pixel_values=images)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("loss:", loss.item())
