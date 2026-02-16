import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class AFHQDataset(Dataset):
    """AFHQ dataset (Animal Faces HQ)"""

    def __init__(self, root, split="train", image_size=64, transform=None):
        self.root = os.path.join(root, "afhq")
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.classes = ["cat", "dog", "wild"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        split_dir = os.path.join(self.root, split)
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    if img_name.endswith((".jpg", ".png", ".jpeg")):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class AFHQDataLoader:
    CLASS_NAMES = ["cat", "dog", "wild"]

    def __init__(
        self, root="../../../Datasets", train=True, image_size=64, device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        split = "train" if train else "val"
        self.dataset = AFHQDataset(
            root=root, split=split, image_size=image_size, transform=self.transform
        )

    def get_samples(self, num_images, step=500):
        """Get sample images and labels from the dataset."""
        images = []
        labels = []
        for i in range(num_images):
            idx = min(i * step, len(self.dataset) - 1)
            img, label = self.dataset[idx]
            images.append(img)
            labels.append(self.CLASS_NAMES[label])
        images = torch.stack(images).to(self.device)
        return images, labels

    def get_dataloader(
        self, batch_size=32, shuffle=True, num_workers=2, pin_memory=False
    ):
        """Get a PyTorch DataLoader for batched iteration."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


if __name__ == "__main__":
    loader = AFHQDataLoader(root="../../../Datasets", train=True, image_size=64)
    print(f"Total samples: {len(loader.dataset)}")
    images, labels = loader.get_samples(num_images=4)
    print(f"Sample shape: {images.shape}")
    print(f"Sample labels: {labels}")
    dataloader = loader.get_dataloader(batch_size=8, num_workers=4)
    batch = next(iter(dataloader))
    imgs, lbls = batch
    print(f"Batch shape: {imgs.shape}")
    print(f"Labels shape: {lbls.shape}")
