import os
import cv2
import numpy as np
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib
# from augmentation import gen_transformations_params
from tqdm.auto import tqdm
from utils import scale_images

# %%
fixed_shape = (240, 240) #--> ogni immagine deve essere di dimensione 240x240
batch_size = 10 # read and preprocess the data in batches

def fit_min_max_scaler(images, min_val, max_val):
    """
    Questa funzione prende un elenco di immagini ridimensionate e fa il fit del MinMaxScaler su di esse.
    """
    flat_images = [image.reshape(-1, 1) for image in images]
    all_pixels = np.concatenate(flat_images, axis=0)
    
    scaler = MinMaxScaler(feature_range=(0, 255)).fit(all_pixels)
    return scaler


# %%

def do_load(file):
    nib_imag = nib.load(file)
    nib_imag_canonical = nib.as_closest_canonical(nib_imag)
    img = nib_imag_canonical.get_fdata()
    img = np.rot90(img, axes=(0, 1))

    if img.shape[:-1] != (512,512):
        print(f"Image {file} shape: {img.shape}")
    return img

def load_all_images(kits_path, folders, fixed_shape, dir_path):
    raw_images = []  
    all_images = []
    clip_range = (-1000, 1000)

    for subfolder in tqdm(folders[362:], total=len(folders), desc='Loading images and masks'):
        
        subfolder_path = os.path.join(kits_path, subfolder)
        img_files = sorted(glob(os.path.join(subfolder_path, '*imaging*.nii.gz')))
        mask_files = sorted(glob(os.path.join(subfolder_path, 'instances/tumor*.nii.gz')))

        if len(mask_files) == 0:
            print(f'No mask files found for {subfolder}')
            continue

        assert len(img_files) == 1

        img_file = img_files[0]
        img = do_load(img_file)
        raw_images.append(img)

        mask_1_files = [mask_file for mask_file in mask_files if 'instance-1' in mask_file]
        mask_2_files = [mask_file for mask_file in mask_files if 'instance-2' in mask_file]
        if len(mask_2_files) == 0 or len(mask_1_files) == 0:
            continue
        mask_ann_1 = np.stack([do_load(mask_file) for mask_file in mask_files if 'instance-1' in mask_file], axis=0)
        mask_ann_2 = np.stack([do_load(mask_file) for mask_file in mask_2_files], axis=0)


        assert mask_ann_1.shape == mask_ann_2.shape
        number_annotations = mask_ann_1.shape[0]

        mask = mask_ann_1 + mask_ann_2
        mask = np.sum(mask, axis=0)
        mask[mask < number_annotations] = 0
        mask[mask == number_annotations] = 1


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


            assert img.shape[0] == img.shape[1]
            scale = fixed_shape[1] / img.shape[1]

            # resize the image and mask to fixed shape
            img = np.stack([cv2.resize(img[...,i], None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4) for i in range(img.shape[-1])], axis=2)
            mask = np.stack([cv2.resize(mask[...,i], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for i in range(mask.shape[-1])], axis=2)

        mask_mask_slice = (mask != 0).any(axis=0).any(axis=0)

        idxs_mask = np.where(mask_mask_slice)[0]
        if len(idxs_mask) >= 40:
            idxs_mask = idxs_mask[np.abs(np.mean(idxs_mask) - idxs_mask) <= 20]

        clip_percentile = (-1000, np.percentile(img, 95))
        img = np.clip(img, clip_percentile[0], clip_percentile[1])

        img = img[..., idxs_mask]
        mask = mask[..., idxs_mask]

        clip_range = (-1000, 1000)
        img = np.clip(img, clip_range[0], clip_range[1])
        img = scale_images(img, *clip_range)

        for i in range(img.shape[2]):
            img_slice = img[..., i]
            mask_slice = mask[..., i]
            img_slice = img_slice.astype(np.float32)
            mask_slice = mask_slice.astype(np.float32)

            assert not np.all(mask_slice == 0), f"Img {img_file} , slice {i} has only 0 mask values"

            all_images.append(img_slice)

            os.makedirs(os.path.join(dir_path, f'{subfolder}'), exist_ok=True)
            np.savez(os.path.join(dir_path, f'{subfolder}', f'{subfolder}_{idxs_mask[i]}'), img=img_slice)
            np.savez(os.path.join(dir_path, f'{subfolder}', f'{subfolder}_{idxs_mask[i]}_mask'), mask=mask_slice)

    return len(all_images)

# %%
kits_path = '../Datasets/kits23/dataset'
output_kits_path = '../Datasets/kits23_kidney_preprocessed_noblack'


if not os.path.exists(output_kits_path):
    os.makedirs(output_kits_path)
folders = [f for f in os.listdir(kits_path) if os.path.isdir(os.path.join(kits_path, f))] # lista dei case_id


n_samples = load_all_images(kits_path, folders, fixed_shape, output_kits_path)
print('n_samples', n_samples)

