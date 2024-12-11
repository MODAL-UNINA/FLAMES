# %%
import os
import cv2
import pickle as pkl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

fixed_shape = (240, 240) 
batch_size = 10 

num_augmentation = 50
seed = 42


def preprocess_breast_ultrasound(df, path, output_preprocessed_path, fixed_shape):
    
    imgs = df['Images'].values
    masks = df['Masks'].values

    total_black_pixels = 0
    total_white_pixels = 0

    all_images_preprocessed = []
    raw_images = []

    imgs_to_delete = pkl.load(open('../Datasets/Breast_US_images_to_delete.pkl', 'rb'))

    for image_name, mask_name in zip(imgs, masks):
        img_type = image_name.split(' ')[0]
        if image_name.split('.')[0] in imgs_to_delete:
            continue

    
        if not os.path.exists(os.path.join(output_preprocessed_path, img_type)):
            os.makedirs(os.path.join(output_preprocessed_path, img_type))

        img = plt.imread(os.path.join(path, f'{img_type}', image_name))
        mask = plt.imread(os.path.join(path, f'{img_type}', mask_name))

        assert img.shape[:2] == mask.shape[:2], f'Image {image_name} and mask {mask_name} must have the same shape'

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
            

        all_images_preprocessed.append(img)
        total_black_pixels += np.sum(mask == 0)
        total_white_pixels += np.sum(mask == 1)

    print('Total black pixels:', total_black_pixels)
    print('Total white pixels:', total_white_pixels)
    print('Total imgs to delete (duplicated)', len(imgs_to_delete))
    print('Number of processed images:', len(all_images_preprocessed))

    
    return len(all_images_preprocessed)


# %% Richiama il preprocessing
path = '../Datasets/Breast_ultrasound/Dataset_BUSI_with_GT'
malignant_path = '../Datasets/Breast_ultrasound/Dataset_BUSI_with_GT/malignant.csv'
benign_path = '../Datasets/Breast_ultrasound/Dataset_BUSI_with_GT/benign.csv'
normal_path = '../Datasets/Breast_ultrasound/Dataset_BUSI_with_GT/normal.csv'
breast_preprocessed_path = '../Datasets/Breast_ultrasound_preprocessed_noblack'

malignant = pd.read_csv(malignant_path)
benign = pd.read_csv(benign_path)
normal = pd.read_csv(normal_path)
malignant_and_benign = pd.concat([malignant, benign], ignore_index=True)

n_samples = preprocess_breast_ultrasound(malignant_and_benign, path, breast_preprocessed_path, fixed_shape)
print('n_samples:', n_samples)
