from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, RandomCrop, Compose, Normalize
from pathlib import Path
from PIL import Image, ImageOps
import torch


class RSDataset(Dataset):
    def __init__(self, dataset_path, device='cuda'):
        self.dataset_path = Path(dataset_path)
        self.image_files = tuple(self.dataset_path.rglob('*.tif'))
        self.images = self.load_images(device)
        self.transform = Compose([
            RandomCrop(224),
            Normalize(mean=[0.4841, 0.4899, 0.4504], std=[0.2181, 0.2022, 0.1959]),
        ])
        self.dict = {'agricultural': 0, 'airplane': 1, 'baseballdiamond': 2, 'beach': 3, 'buildings': 4, 'chaparral': 5,
                     'denseresidential': 6, 'forest': 7, 'freeway': 8, 'golfcourse': 9, 'harbor': 10,
                     'intersection': 11, 'mediumresidential': 12, 'mobilehomepark': 13, 'overpass': 14,
                     'parkinglot': 15, 'river': 16, 'runway': 17, 'sparseresidential': 18, 'storagetanks': 19,
                     'tenniscourt': 20}

    def __len__(self):
        return len(self.images)

    def load_images(self, device):
        images = []
        for img_file in self.image_files:
            img = Image.open(img_file)
            if img.size != (256, 256):
                img = ImageOps.pad(img, (256, 256))
            img = ToTensor()(img)
            images.append(img)
        return torch.stack(images).to(device)

    def __getitem__(self, item):
        image = self.transform(self.images[item])
        label = self.dict[self.image_files[item].stem[:-2]]
        return image, label
