import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

class Convlayer(nn.Module):
    def __init__(self):
        super(Convlayer, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.lrelu = nn.LeakyReLU(0.01)
        
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.bn_out = nn.BatchNorm2d(1)
        
    def forward(self, x):
        residual = x  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        
        x = self.conv_out(x)
        x = self.bn_out(x)
        
        x = x + residual
        x = self.lrelu(x)
        
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = Convlayer()  
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, gt, refs, thresh=0.5):
        gt = self.conv(gt)
        gt_feat = self.cnn(gt)
        batch_size, c, h, w = gt_feat.shape
        # Shape: [batch_size, c, h, w]
        
        # refs_feat has shape: [num_refs, batch_size, c, h, w]
        refs_feat = torch.stack([self.cnn(ref) for ref in refs], dim=0)
        # print("gt_feat min:", gt_feat.min().item(), "max:", gt_feat.max().item())
        # print("refs_feat min:", refs_feat.min().item(), "max:", refs_feat.max().item())

        # Shape: [num_refs, batch_size, c, h, w]
       
        # gt_feat_expanded shape: [batch_size, 1, c, h, w]
        gt_feat_expanded = gt_feat.unsqueeze(1)
        # Shape: [batch_size, 1, c, h, w]
        
        # diff_maps shape: [batch_size, num_refs, c, h, w]
        diff_maps = torch.abs(gt_feat_expanded - refs_feat)
        # Shape: [batch_size, num_refs, c, h, w]
        
        # Sum the differences across channels
        # Shape: [batch_size, num_refs, h, w]
        total_dissim_map = diff_maps.sum(dim=2)

        # Min across all reference images (num_refs) for each batch, for every pixel (h, w)
        min_dissim = total_dissim_map.min(dim=1, keepdim=True)[0]  # Min across num_refs
        min_dissim = min_dissim.view(batch_size, 1, h, w)  # Shape: [batch_size, 1, h, w]
        
        # Max across all reference images (num_refs) for each batch, for every pixel (h, w)
        max_dissim = total_dissim_map.max(dim=1, keepdim=True)[0]  # Max across num_refs
        max_dissim = max_dissim.view(batch_size, 1, h, w)  # Shape: [batch_size, 1, h, w]

        normalized_dissim_map = (total_dissim_map - min_dissim) / (max_dissim - min_dissim + 1e-8)
        # Shape: [batch_size, num_refs, h, w]
        
        normalized_dissim_map = torch.clamp(normalized_dissim_map, min=0.1, max=1.0)
        
        # Thresholding
        dissim_map = torch.where(normalized_dissim_map > thresh, normalized_dissim_map, torch.zeros_like(normalized_dissim_map))
        # Shape: [batch_size, num_refs, h, w]
        
        # Find the index of the maximum dissimilarity value per pixel
        _, ref_idx_map = torch.max(dissim_map, dim=1)
        # ref_idx_map Shape: [batch_size, h, w]
        
        ref_idx_map = ref_idx_map.unsqueeze(1)
        # Shape: [batch_size, 1, h, w]
        
        collage = torch.zeros((batch_size, 1, h, w), device=gt.device)
        # print(len(refs[1])) = 8
        # Shape: [batch_size, 1, h, w]
        for ref_idx in range(len(refs[1])):
            ref_img = refs_feat[:, ref_idx, :, :, :]
            # Shape: [batch_size, c, h, w]
            
            mask = (ref_idx_map == ref_idx)
            # Shape: [batch_size, 1, h, w]
            
            collage = torch.where(mask, ref_img, collage)
            # collage += mask * ref_img
            
        # Shape: [batch_size, 1, h, w]
        return dissim_map, collage

def view_output(output, gt):
    batch_size = gt.size(0)  # Get the batch size
    for i in range(batch_size):
        # Convert ground truth to grayscale for the current sample in the batch
        gt_gray = gt[i].mean(dim=0, keepdim=True)  # Shape: [1, h, w]
        gt_np = gt_gray.squeeze().cpu().detach().numpy()  # Shape: [h, w]

        collage_np = output[i].cpu().detach().numpy()  # Shape: [3, h, w] or [1, h, w]

        if collage_np.shape[0] == 1:
            # If the collage is grayscale, convert it to RGB for overlay
            collage_rgb = np.stack([collage_np.squeeze()] * 3, axis=0)  # Shape: [3, h, w]
        elif collage_np.shape[0] == 3:
            # If the collage is already RGB, use it directly
            collage_rgb = collage_np
        else:
            raise ValueError("Collage should have 1 or 3 channels.")

        # Normalize the collage_rgb to the range [0, 1]
        collage_rgb = (collage_rgb - collage_rgb.min()) / (collage_rgb.max() - collage_rgb.min() + 1e-8)

        # Normalize the ground truth image to [0, 1]
        gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

        # Create a blue overlay based on the collage
        blue_overlay = np.zeros_like(collage_rgb)  # Shape: [3, h, w]
        blue_channel = collage_rgb[2]  # Blue channel
        blue_overlay[2] = blue_channel  # Set the blue channel

        # Normalize the blue overlay to [0, 1]
        max_val = blue_overlay.max()
        if max_val > 0:
            blue_overlay /= max_val  # Normalize
        else:
            blue_overlay = np.zeros_like(blue_overlay)

        # Create a grayscale ground truth with 3 channels for overlay
        gt_rgb = np.stack([gt_np] * 3, axis=0)  # Shape: [3, h, w]

        # Combine ground truth and blue overlay
        combined_image = np.clip(gt_rgb + blue_overlay, 0, 1)  # Add blue overlay to the grayscale ground truth

        # Plot the images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(collage_rgb.transpose(1, 2, 0), cmap='gray')  # Convert [3, h, w] to [h, w, 3] for imshow
        plt.title(f'Collaged Output')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gt_np, cmap='gray')
        plt.title(f'Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(combined_image.transpose(1, 2, 0))  # Convert [3, h, w] to [h, w, 3] for imshow
        plt.title('Ground Truth with Blue Overlay')
        plt.axis('off')

        plt.show()

def view_extract(data_loader, model, device):
    for batch_idx, (gt, refs) in enumerate(data_loader):
        ground_truth, references = gt.to(device), [ref.to(device) for ref in refs]
        _, collage = model(ground_truth, references)
        view_output(collage, ground_truth)
        torch.cuda.empty_cache()


