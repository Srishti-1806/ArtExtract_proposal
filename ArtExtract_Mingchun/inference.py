import torch
import warnings
warnings.filterwarnings("ignore")

from utils.visulization import extract_hidden_art
from utils.data_graph import load_inference_datasets
from model.extract_model import GATSiameseNetwork

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    feature_dim = 14  # Example feature dimension, adjust as needed
    model = GATSiameseNetwork(in_channels=feature_dim, hidden_channels=128, out_channels=32).to(device) # Ensure the model matches your architecture
    
    # Load pre-trained model weights
    model.load_state_dict(torch.load('./checkpoints/GAT/best_model.pth', map_location=device))
    
    # Load validation dataset
    val_loader = load_inference_datasets('./dataset/val', batch_size=1)
    
    # Extract and visualize hidden art features
    extract_hidden_art(model, val_loader, device, save_dir='./img', mode='diff', alpha=0.5)

if __name__ == "__main__":
    main()