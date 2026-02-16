import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TinyImageNetDataset(Dataset):
    """Tiny ImageNet dataset (200 classes, 64×64)"""

    def __init__(self, root, split="train", transform=None):
        self.root = os.path.join(root, "tiny-imagenet-200")
        self.split = split
        self.transform = transform

        self.images = []
        self.labels = []

        if split == "train":
            train_dir = os.path.join(self.root, "train")
            # Get class names and create mapping
            self.classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            for class_name in self.classes:
                class_dir = os.path.join(train_dir, class_name, "images")
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

        elif split == "val":
            val_dir = os.path.join(self.root, "val")
            # Read val annotations
            with open(os.path.join(val_dir, "val_annotations.txt"), "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_name = parts[0]
                    class_name = parts[1]
                    self.images.append(os.path.join(val_dir, "images", img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class TinyImageNetDataLoader:
    def __init__(self, root="../../../Datasets", train=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        split = "train" if train else "val"
        self.dataset = TinyImageNetDataset(
            root=root, split=split, transform=self.transform
        )

        # Load class names from words.txt
        words_file = os.path.join(root, "tiny-imagenet-200", "words.txt")
        self.CLASS_NAMES = []
        if os.path.exists(words_file):
            with open(words_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        self.CLASS_NAMES.append(parts[1])

        # Fallback if words.txt not available
        if not self.CLASS_NAMES:
            self.CLASS_NAMES = [f"class_{i}" for i in range(200)]

    def get_samples(self, num_images, step=500):
        """Get sample images and labels from the dataset."""
        images = []
        labels = []
        for i in range(num_images):
            img, label = self.dataset[i * step]
            images.append(img)
            labels.append(
                self.CLASS_NAMES[label]
                if label < len(self.CLASS_NAMES)
                else f"class_{label}"
            )
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
    loader = TinyImageNetDataLoader(root="../../../Datasets", train=True)
    print(f"Total samples: {len(loader.dataset)}")
    images, labels = loader.get_samples(num_images=4)
    print(f"Sample shape: {images.shape}")
    print(f"Sample labels: {labels}")
    dataloader = loader.get_dataloader(batch_size=8, num_workers=4)
    batch = next(iter(dataloader))
    imgs, lbls = batch
    print(f"Batch shape: {imgs.shape}")
    print(f"Labels shape: {lbls.shape}")
