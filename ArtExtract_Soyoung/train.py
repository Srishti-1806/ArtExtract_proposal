import os
import gc
import wandb
import torch
import warnings
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torch.cuda.amp as amp 
from torchvision import transforms, models
from utils.data import load_datasets
from model import SimplyUNet
from utils.vizImg import plot_images, viz_train
from utils.metrics import MS_SSIMLoss, EvalMetrics
from torchvision.models.vgg import VGG16_Weights
# # For the comparative experiements (Optional)
# from unets.baseUnet import BaseUNet
# from unets.sertUnet import SERTUnet
# from unets.sparseUnet import SparseUNet

warnings.filterwarnings("ignore")

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        vgg.features[0] = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.features = vgg.features[:23] 
        
    def forward(self, x):
        return self.features(x)

def train_test_model(model, train_path, val_path, optimizer, scheduler, device, epochs, project_name, entity_name, run_name):
    # Initialize W&B
    wandb.init(project=project_name, entity=entity_name,name=run_name,dir=None) # Disable local logging
    
    vgg_feature_extractor = VGGFeatureExtractor().to(device, non_blocking=True)
    ms_ssim_loss = MS_SSIMLoss().to(device)
    metrics = EvalMetrics(vgg_feature_extractor).to(device, non_blocking=True)
    train_loader, val_loader = load_datasets(train_path, val_path)
    
    best_val_lpips = float('inf')
    best_model_dir = "best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_number = 0
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss, epoch_psnr, epoch_lpips, epoch_ssim = 0.0, 0.0, 0.0, 0.0
        optimizer.zero_grad()
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            output = model(images)
            masks = masks.squeeze(2)    
            assert masks.shape == output.shape  # [B, 8, H, W]
            output = F.normalize(output, dim=1)
            masks = F.normalize(masks, dim=1)
            # (Optional) If you want to visually check the image.
            # if epoch % 3 == 0: 
            #     plot_images(output, masks, epoch) 
            loss = ms_ssim_loss(output, masks) / 2  # Adjust for accumulation steps
            loss.backward()
            
            if (step + 1) % 2 == 0:  # Adjust for accumulation steps
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            total_loss += loss.item()
            psnr_value, lpips_value, ssim_value = metrics(output.detach(), masks.detach())
            
            epoch_psnr += psnr_value.item()
            epoch_lpips += lpips_value.item()
            epoch_ssim += ssim_value.item()

            del images, masks, output, loss
            del psnr_value, lpips_value, ssim_value
            torch.cuda.empty_cache()
            gc.collect()
            
        average_loss = total_loss / len(train_loader)
        epoch_psnr /= len(train_loader)
        epoch_lpips /= len(train_loader)
        epoch_ssim /= len(train_loader)
        
        # Evaluate the model on the validation set
        val_psnr, val_lpips, val_ssim = evaluate_model(model, val_loader, device, metrics)
        
        wandb.log({
            'epoch': epoch + 1,
            'loss': average_loss,
            'train_psnr': epoch_psnr,
            'train_lpips': epoch_lpips,
            'train_ssim': epoch_ssim,
            'val_psnr': val_psnr,
            'val_lpips': val_lpips,
            'val_ssim': val_ssim
        })

        # Save the best model based on validation LPIPS
        # if val_lpips < best_val_lpips:
        #     best_val_lpips = val_lpips
        #     new_model_path = os.path.join(best_model_dir, f"baseUnet_{best_model_number}_{best_val_lpips:.4f}.pth")
        #     if best_model_number > 0:
        #         old_model_path = os.path.join(best_model_dir, f"baseUnet_{best_val_lpips:.4f}.pth")
        #         if os.path.exists(old_model_path):
        #             os.remove(old_model_path)
        #     torch.save(model.state_dict(), new_model_path)
        #     best_model_number += 1
        #     print(f"Best model saved with Validation LPIPS: {best_val_lpips:.4f}")
        
        del epoch_psnr, epoch_lpips,epoch_ssim, val_psnr,val_lpips,val_ssim
        torch.cuda.empty_cache()
        gc.collect() 

    del vgg_feature_extractor, ms_ssim_loss, metrics, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

def evaluate_model(model, val_loader, device, metrics):
    model.eval()
    val_psnr, val_lpips, val_ssim = 0.0, 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True) 
            batch_size, num_channels = images.size(0), images.size(1)
            output = model(images)  
            masks = masks.squeeze(2)  
            assert masks.shape == output.shape
            
            # Normalize the entire batch
            output = F.normalize(output, dim=1)
            masks = F.normalize(masks, dim=1)
                
            # Calculate metrics for the batch
            psnr_value, lpips_value, ssim_value = metrics(output.detach(), masks.detach())
            total_samples += batch_size 
            val_psnr += psnr_value.item()
            val_lpips += lpips_value.item()
            val_ssim += ssim_value.item()
            torch.cuda.empty_cache()
            del images, masks, output
            
    num_batches = len(val_loader)
    val_psnr /= num_batches
    val_lpips /= num_batches
    val_ssim /= num_batches
    torch.cuda.empty_cache()
    return val_psnr, val_lpips, val_ssim

def gen_img(model, best_model_path, test_path, output_dir, device):
    model.load_state_dict(torch.load(best_model_path))
    model.to(device, non_blocking=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor(),   
    ])
    
    os.makedirs(output_dir,exist_ok=True)
    for img_name in os.listdir(test_path):
        img_path = os.path.join(test_path, img_name)
 
        if os.path.isfile(img_path):
            # Open and transform the image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device, non_blocking=True) 
            with torch.no_grad():
                output = model(img_tensor)

            for j in range(output.size(1)): 
                output_channel = output[:, j, :, :] 
                output_channel_np = output_channel.cpu().detach().numpy()
                
                plt.figure(figsize=(5, 5))
                plt.imshow(output_channel_np[0], cmap='gray')
                plt.title(f'{os.path.splitext(img_name)[0]}_channel_{j}')
                plt.axis('off')
                
                channel_img_name = f"{os.path.splitext(img_name)[0]}_channel_{j}.png"
                plt.savefig(os.path.join(output_dir, channel_img_name), bbox_inches='tight', pad_inches=0)
                plt.close()
    print('Successfully saved image.')

def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a U-Net model")
    parser.add_argument('-tr', '--trainpath', type=str, required=True, help='Path to training images')
    parser.add_argument('-v', '--valpath', type=str, required=True, help='Path to validation images')
    parser.add_argument('-te', '--testpath', type=str, help='Path to test images for generating multispectral images')  
    parser.add_argument('-lr', '--learningRate', type=float, default=0.02, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('-g', '--genimg', action='store_true', help='Generate multispectral images')
    parser.add_argument('--project', type=str, default='GSOC', help='W&B project name')
    parser.add_argument('--name', type=str, default='run1', help='W&B run name')
    # Put your entity name at the default section!
    parser.add_argument('--entity', type=str, default='ENTER YOUR ENTITY NAME', help='W&B entity name')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimplyUNet().to(device, non_blocking=True) 
    learning_rate = args.learningRate
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    train_test_model(
        model, 
        args.trainpath, 
        args.valpath, 
        optimizer, 
        scheduler, 
        device, 
        epochs=args.epochs,
        project_name=args.project, 
        entity_name=args.entity,
        run_name = args.name
    )
    
    if args.genimg:
        gen_img(model, args.best_model_path, args.testpath, args.outputpath, device)
        print('Image generation completed')
