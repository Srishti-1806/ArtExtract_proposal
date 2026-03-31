import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from extract.data import load_datasets
from extract.extract import SiameseNetwork

def train(model, device, train_loader, val_loader, optimizer, epochs):
    model.train() 
    for epoch in range(epochs): 
        train_loss = 0
        for batch_idx, (gt, refs) in enumerate(train_loader):
            ground_truth, references = gt.to(device), [ref.to(device) for ref in refs]
            optimizer.zero_grad()
            norm_dissimilarity, collage = model(ground_truth, references)
            
            # Reverse objective of training -> Maximize dissimilarity
            loss = -norm_dissimilarity.mean()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            train_loss += loss.item()
            print(f'Epoch: {epoch} ({100. * batch_idx / len(train_loader):.0f}%)\tLoss: {loss.item():.6f}') 
            
            # Visualize periodically (e.g., every 5 epochs)
            # if epoch % 5 == 0:
            #     view_output(collage, gt, epoch)
                
        val_loss, _ = validate(model, device, val_loader, epoch)

def validate(model, device, val_loader, epoch):
    model.eval() 
    val_loss = 0
    max_dissimilarity_collage = None

    with torch.no_grad():
        for batch_idx, (gt, refs) in enumerate(val_loader):
            ground_truth, references = gt.to(device), [ref.to(device) for ref in refs]
            norm_dissimilarity, collage = model(ground_truth, references)
            loss = -norm_dissimilarity.mean()
            val_loss += loss.item()
            torch.cuda.empty_cache()
            
            # Optional: Visualize validation results
            if epoch % 5 == 0:
                view_output(collage, gt, epoch)

            # Track the max dissimilarity collage
            if max_dissimilarity_collage is None or loss.item() > val_loss:
                max_dissimilarity_collage = collage

    val_loss /= len(val_loader.dataset)
    print(f'\nVal: Averaged DSSIM loss: {val_loss:.6f}\n')
    
    return val_loss, max_dissimilarity_collage


def view_output(output, gt, epoch):
    # Convert ground truth to grayscale
    gt_gray = gt.mean(dim=1, keepdim=True)  # Shape: [batch_size, 1, h, w]
    gt_np = gt_gray[0].squeeze().cpu().detach().numpy()  # Shape: [h, w]
    collage_np = output[0].cpu().detach().numpy()  # Shape: [3, h, w] or [1, h, w]

    if collage_np.shape[0] == 1:
        # If the collage is grayscale, convert it to RGB for overlay
        collage_rgb = np.stack([collage_np.squeeze()] * 3, axis=0)  # Shape: [3, h, w]
    elif collage_np.shape[0] == 3:
        # If the collage is already RGB, use it directly
        pass
    else:
        raise ValueError("Collage should have 1 or 3 channels.")

    blue_overlay = np.zeros_like(collage_rgb)  # Shape: [3, h, w]
    blue_channel = collage_rgb[2]  # Blue channel
    blue_overlay[2] = blue_channel  # Set the blue channel

    # Normalize the blue overlay to [0, 1] for visualization
    max_val = blue_overlay.max()
    if max_val > 0:
        blue_overlay /= max_val  # Normalize
    else:
        blue_overlay = np.zeros_like(blue_overlay)
    gt_rgb = np.stack([gt_np] * 3, axis=0)
    combined_image = np.clip(gt_rgb + blue_overlay, 0, 1)  # Add blue overlay to the grayscale ground truth
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(collage_np.transpose(1, 2, 0),cmap='gray')  # Convert [3, h, w] to [h, w, 3] for imshow
    plt.title(f'Collage - Epoch {epoch}')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gt_np, cmap='gray')
    plt.title(f'GT - Epoch {epoch}')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(combined_image.transpose(1, 2, 0))  # Convert [3, h, w] to [h, w, 3] for imshow
    plt.title('GT with Collage Overlay')
    plt.axis('off')
    
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a U-Net model")
    parser.add_argument('-batch', '--batch', type=str, required=False,default=8, help='batchsize')
    parser.add_argument('-tr', '--trainpath', type=str, required=True, help='Path to training images')
    parser.add_argument('-v', '--valpath', type=str, required=True, help='Path to validation images')
    parser.add_argument('-te', '--testpath', type=str, help='Path to test images for generating multispectral images')  
    parser.add_argument('-lr', '--learningRate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='Number of epochs')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SiameseNetwork().to(device)
    learning_rate = args.learningRate
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    train_loader, val_loader = load_datasets(args.trainpath, args.valpath,args.batch)
    train(model, device, train_loader, val_loader, optimizer, args.epochs)
    print('Train/Validation completed')