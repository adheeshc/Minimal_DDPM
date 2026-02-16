import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CelebADataLoader:
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

        split = "train" if train else "valid"
        self.dataset = datasets.CelebA(
            root=root,
            split=split,
            download=False,
            transform=self.transform,
        )

        # CelebA is unconditional (just faces), so no class names
        self.CLASS_NAMES = ["face"]

    def get_samples(self, num_images, step=1000):
        """Get sample images from the dataset."""
        images = []
        labels = []
        for i in range(num_images):
            idx = min(i * step, len(self.dataset) - 1)
            img, _ = self.dataset[idx]
            images.append(img)
            labels.append("face")
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
    loader = CelebADataLoader(root="../../../Datasets", train=True, image_size=64)
    print(f"Total samples: {len(loader.dataset)}")
    images, labels = loader.get_samples(num_images=4)
    print(f"Sample shape: {images.shape}")
    print(f"Sample labels: {labels}")
    dataloader = loader.get_dataloader(batch_size=8, num_workers=4)
    batch = next(iter(dataloader))
    imgs, lbls = batch
    print(f"Batch shape: {imgs.shape}")
    print(f"Labels shape: {lbls.shape}")
