# %%
import os
import cv2
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from utils import analyze_image_values

# %%

def open_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img

def preprocess_OTU(otu_path, otu_preprocessed_path, fixed_shape):
    
    list_images = os.listdir(otu_path + '/images')
    list_masks = os.listdir(otu_path + '/annotations')
    list_masks = [mk for mk in list_masks if mk.endswith('_binary.PNG') and not mk.endswith('binary_binary.PNG')]
    print(f'Number of images: {len(list_images)}')
    print(f'Number of masks: {len(list_masks)}')

    list_images.sort(key=lambda x: int(x.split('.')[0]))
    list_masks.sort(key=lambda x: int(x.split('_')[0]))

    if not os.path.exists(otu_preprocessed_path + '/images'):
        os.makedirs(otu_preprocessed_path + '/images')
    if not os.path.exists(otu_preprocessed_path + '/annotations'):
        os.makedirs(otu_preprocessed_path + '/annotations')

    total_zeros = 0
    total_ones = 0

    processed_images = []
    raw_images = [] 

    for i, (img_name, mk_name) in tqdm(enumerate(zip(list_images, list_masks)), total=len(list_images)):
        img = open_image(otu_path + '/images/' + img_name)

        mask = open_image(otu_path + '/annotations/' + mk_name)

        if np.unique(mask).shape[0] == 1 and np.unique(mask)[0] == 0:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img / 255.0
        raw_images.append(img) 

        if img.shape[:2] != fixed_shape:
            if img.shape[0] != img.shape[1]:
                if img.shape[0] > img.shape[1]:
                    full_pad = img.shape[0] - img.shape[1]
                    half_pad = full_pad // 2
                    padx = (0, 0)
                    pady = (half_pad + full_pad % 2, half_pad)
                else:
                    full_pad = img.shape[1] - img.shape[0]
                    half_pad = full_pad // 2
                    padx = (half_pad + full_pad % 2, half_pad)
                    pady = (0, 0)
                img = cv2.copyMakeBorder(img, *padx, *pady, cv2.BORDER_CONSTANT, value=0)
                mask = cv2.copyMakeBorder(mask, *padx, *pady, cv2.BORDER_CONSTANT, value=0)

            scale = fixed_shape[0] / img.shape[0]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            img = np.clip(img, 0, 1)
        
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        processed_images.append(img)

        total_zeros += np.sum(mask == 0)
        total_ones += np.sum(mask == 1)

        np.savez(otu_preprocessed_path + '/images/' + str(i + 1), img=img)
        np.savez(otu_preprocessed_path + '/annotations/' + str(i + 1), mask=mask)

                 
    print('Analyzing raw image pixel values...')
    analyze_image_values(raw_images, 'Pixel Value Distribution OTU - Raw Images')

    print('Analyzing processed image pixel values...')
    analyze_image_values(processed_images, 'Pixel Value Distribution OTU - Processed Images')

    print(f'Total number of 0s in masks: {total_zeros}, percentage: {total_zeros / (total_zeros + total_ones) * 100:.2f}%')
    print(f'Total number of 1s in masks: {total_ones}, percentage: {total_ones / (total_zeros + total_ones) * 100:.2f}%')

    return len(processed_images)

# %%
fixed_shape = (240, 240)


# OTU_2d
otu_path = '../Datasets/OTU_2d'
otu_preprocessed_path = '../Datasets/OTU_2d_preprocessed_noblack'

n_samples = preprocess_OTU(otu_path, otu_preprocessed_path, fixed_shape)
print('n_samples', n_samples)

