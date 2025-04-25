import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

def visualise(images, mode="denoise"):
    if mode == "denoise":
        num_images = len(images)
        for i in range(0, num_images, 9):
            current_images = images[i : i + 9]
            f, axes = plt.subplots(3, 3, figsize=(9, 9))
            for ax, img in zip(axes.flat, current_images):
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            plt.show()
    elif mode == "interpolate":
        num_images = len(images)
        for i in range(0, num_images, 24):
            current_images = images[i : i + 24]
            f, axes = plt.subplots(3, 8, figsize=(16, 6))
            for ax, img in zip(axes.flat, current_images):
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            plt.show()
    elif mode == "pairs":
        num_images = len(images)
        for i in range(0, num_images, 2):
            current_images = images[i : i + 2]
            f, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, img in zip(axes.flat, current_images):
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            plt.show()


def test(mode="denoise"):
    images = []
    model_weights = "Lab 1/MLP.8.pth"
    input_dims = 28 * 28
    bottleneck_size = 8

    model = autoencoderMLP4Layer(
        N_input=input_dims, N_bottleneck=bottleneck_size, N_output=input_dims
    )

    model.load_state_dict(torch.load(model_weights))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_transform = transforms.Compose([transforms.ToTensor()])

    test_set = MNIST("./data/mnist", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    with torch.no_grad():
        model.eval()
        for i, (img, _) in enumerate(test_loader):
            img = img.view(1, -1).to(device)
            output = model(img)
            noisy_img = add_noise(img).view(1, -1).to(device)
            images.append((img.cpu().view(28, 28), noisy_img.cpu().view(28, 28), output.cpu().view(28, 28)))

    if mode == "denoise":
        visualise([item for triplet in images for item in triplet], mode="denoise")
    elif mode == "interpolate":
        all_interpolated = []
        temp_images = []
        for img, _ in test_loader:
            temp_images.append(img)
            if len(temp_images) == 2:
                decoded = interpolate(model, device, temp_images[0], temp_images[1], steps=8)
                all_interpolated.extend(decoded)
                temp_images.clear() 
        visualise(all_interpolated, mode="interpolate")
    elif mode == "pairs":
        visualise([triplet[i] for triplet in images for i in [0, 2]], mode="pairs")


def add_noise(img):
    noise = torch.randn_like(img)
    noisy_image = img + noise * (0.5)
    return noisy_image

def interpolate(model, device, x, y, steps=8):
    x = x.view(1, -1).to(device)
    y = y.view(1, -1).to(device)

    interpolated = []
    decoded = []

    with torch.no_grad():
        bottleneck = model.double_encode(x, y)

    for i in range(steps):
        interpolated_bottleneck = bottleneck[0] * (1 - (i / steps)) + bottleneck[1] *  (i / steps)
        interpolated.append(interpolated_bottleneck)

    for bottleneck in interpolated:
        with torch.no_grad():
            decoded_image = model.decode(bottleneck)
            decoded.append(decoded_image.cpu().view(28, 28))

    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["denoise", "interpolate", "pairs"], 
                        help="Mode to visualize: 'denoise' or 'interpolate' or 'pairs'.")
    args = parser.parse_args()

    if args.mode:
        test(mode=args.mode)
    else:
        print("No mode provided! Run with the --mode flag! Exiting... ")
        exit()

if __name__ == "__main__":
    main()