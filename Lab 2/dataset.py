import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import re

class PetNosesDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.annotations = self._parse_txt_file(txt_file)
        self.root_dir = root_dir
        self.transform = transform

    def _parse_txt_file(self, txt_file):
        annotations = []
        with open(txt_file, 'r') as file:
            for line in file:
                image_name, nose_coord = line.strip().split('"')[0:2]
                image_name = image_name.strip(',')
                nose_coord = tuple(map(int, re.findall(r'\d+', nose_coord)))
                annotations.append((image_name, nose_coord))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, nose_coord = self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        scale_x = 227 / original_size[0]
        scale_y = 227 / original_size[1]
        scaled_nose_coords = torch.tensor([
            nose_coord[0] * scale_x,
            nose_coord[1] * scale_y
        ])

        if self.transform: image = self.transform(image)
        return image, scaled_nose_coords