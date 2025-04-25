import argparse
import torch
import numpy as np
import time
from dataset import PetNosesDataset
from model import CNN
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def get_transform():
    return v2.Compose(
        [
            v2.Resize((227, 227)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load("snoutnet.pth"))

    transform = get_transform()
    test_dataset = PetNosesDataset(
        txt_file=args.test_txt, root_dir=args.data_path, transform=transform
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model.eval()
    distances = []
    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for images, nose_coords in test_loader:
            images, nose_coords = images.to(device), nose_coords.to(device)
            outputs = model(images)
            batch_distances = torch.sqrt(torch.sum((outputs - nose_coords) ** 2, dim=1))
            distances.extend(batch_distances.cpu().numpy())
            total_images += images.size(0)

    elapsed_time = time.time() - start_time
    avg_time_per_image = elapsed_time / total_images

    distances = np.array(distances)
    print(f"\nLocalization Accuracy Statistics:")
    print(f"Minimum Distance: {distances.min():.4f}")
    print(f"Mean Distance: {distances.mean():.4f}")
    print(f"Maximum Distance: {distances.max():.4f}")
    print(f"Standard Deviation: {distances.std():.4f}")
    print(f"\nAverage Classification Time per Image: {avg_time_per_image:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test CNN model for Pet Nose Localization"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Absolute path to the dataset directory",
    )
    parser.add_argument(
        "--test_txt", type=str, default="./test_noses.txt", help="Path to test txt file"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    main(args)
