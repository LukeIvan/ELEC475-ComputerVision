import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import random  # Import the random module

from model import CNN
from dataset import PetNosesDataset

# Load the dataset
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PetNosesDataset(
    txt_file="./dataset/oxford-iiit-pet-noses/test_noses.txt",
    root_dir="./dataset/oxford-iiit-pet-noses/images-original/images",
    transform=transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)  # Ensure you have your model defined
model.load_state_dict(torch.load("snoutnet.pth"))  # Load your trained model
model.eval()

def visualize_predictions(dataset, model, num_images=9):
    plt.figure(figsize=(12, 12))
    
    # Randomly select num_images indices from the dataset
    indices = random.sample(range(len(dataset)), num_images)  # Randomly sample indices
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, actual_coords = dataset[idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension for model input
            
            # Get predicted coordinates from the model
            predicted_coords = model(image).cpu().numpy().flatten()  # Move to CPU and convert to NumPy

            # Convert the image tensor to a NumPy array
            image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and move to CPU
            image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
            image = (image * 255).astype('uint8')  # Scale to [0, 255]

            plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            
            # Print the actual and predicted coordinates for debugging
            print(f"Image: {dataset.annotations[idx][0]}, Actual Coordinates: {actual_coords.numpy()}, Predicted Coordinates: {predicted_coords}")

            # Scatter plot for actual and predicted coordinates
            plt.scatter(actual_coords[0].item(), actual_coords[1].item(), color='green', s=40, label='Actual')
            plt.scatter(predicted_coords[0], predicted_coords[1], color='red', s=40, label='Predicted')
            plt.axis('off')
            plt.title(f"Actual (green) vs Predicted (red)")
    
    plt.tight_layout()
    plt.legend()
    plt.show()

# Call the visualization function
visualize_predictions(dataset, model, num_images=9)   # Visualize 9 random images
