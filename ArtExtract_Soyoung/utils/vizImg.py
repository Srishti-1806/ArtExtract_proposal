import matplotlib.pyplot as plt
import numpy as np

# Display the output image and compare with target image visually
# Output can be blurry after the normalisation
def plot_images(output_image, target_image, epoch):
    # Fixed to view 1st image in batch, 3rd channel
    output_image = output_image[0, 2]
    target_image = target_image[0, 2]
    output_image_np = output_image.squeeze().cpu().detach().numpy()
    target_image_np = target_image.squeeze().cpu().detach().numpy()
    plt.figure(figsize=(7, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(output_image_np, cmap='gray')
    plt.title(f'Output - Epoch {epoch}',fontsize=9)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(target_image_np, cmap='gray')
    plt.title(f'Target - Epoch {epoch}',fontsize=9)
    plt.axis('off')
    
    plt.show()

# If you are not using W$B, you can manually plot the training progress
def viz_train(losses,train_lpipses, val_lpipses,train_psnrs,val_psnrs,train_ssims,val_ssims):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting PSNRs
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_lpipses) + 1), train_lpipses, label='Train LPIPS')
    plt.plot(np.arange(1, len(val_lpipses) + 1), val_lpipses, label='Val LPIPS')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.title('Training and Validation LPIPS')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_psnrs) + 1), train_psnrs, label='Train PSNR')
    plt.plot(np.arange(1, len(val_psnrs) + 1), val_psnrs, label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training and Validation PSNR')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_ssims) + 1), train_ssims, label='Train SSIM')
    plt.plot(np.arange(1, len(val_ssims) + 1), val_ssims, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training and Validation SSIM')
    plt.legend()
    plt.show()