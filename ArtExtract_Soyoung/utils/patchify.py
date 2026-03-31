import os
from PIL import Image

def save_patches(img, save_dir, base_name, isImg=False):
    w, h = img.size
    half_w, half_h = w // 2, h // 2

    boxes = [
        (0, 0, half_w, half_h),           # Top-left
        (half_w, 0, w, half_h),           # Top-right
        (0, half_h, half_w, h),           # Bottom-left
        (half_w, half_h, w, h),           # Bottom-right
    ]

    for i, box in enumerate(boxes):
        image_patch = img.crop(box)
        if isImg:
             image_patch.save(os.path.join(save_dir, f'{base_name}_p{i}_RGB.png'))
        else:
            prefix = base_name.split('_ms')[0]
            suffix = base_name.split('_ms')[1]
            image_patch.save(os.path.join(save_dir, f'{prefix}_p{i}{suffix}_ms.png'))


# Directories
images_dir = '../train/rgb_images' 
masks_dir = '../train/ms_masks'
patches_img = '../trainPatch/rgb_images'
patches_ms = '../trainPatch/ms_masks'

# Ensure patch directories exist
os.makedirs(patches_img, exist_ok=True)
os.makedirs(patches_ms, exist_ok=True)

# Iterate over all images
for img_name in os.listdir(images_dir):
    if img_name.endswith(('RGB.bmp', '.png', '.jpg')):
        img_path = os.path.join(images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        filename, ext = os.path.splitext(img_name)
        base_name = filename.replace('_RGB', '')
        save_patches(image, patches_img, base_name, True)

        # Load corresponding masks
        mask_names = sorted([f for f in os.listdir(masks_dir) if f.startswith(base_name)])
        print('mask_names:'.len(mask_names))
        for mask_name in mask_names:
            mask_path = os.path.join(masks_dir, mask_name)
            mask = Image.open(mask_path)
            save_patches(mask, patches_ms, mask_name.split('.')[0])  # Use mask name for unique identification
