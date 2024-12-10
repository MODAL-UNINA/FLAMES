import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import load_niib_data, scale_images

# %%
# read csv
path = '../Datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
df_orig = pd.read_csv(os.path.join(path, 'name_mapping.csv'))

df_2017 = df_orig[['BraTS_2017_subject_ID', 'BraTS_2020_subject_ID']]
df_2017 = df_2017[df_2017['BraTS_2017_subject_ID'].isnull()]


# %%
output_path = '../Datasets/BraTS2020_preprocessed_noblack'


raw_imgs = []
processed_imgs = []
for patient in tqdm(os.listdir(path), desc='Patients', total=len(os.listdir(path))):
    if not patient.startswith('BraTS20'):
        continue
    if patient in df_2017['BraTS_2020_subject_ID'].values:
        img = load_niib_data(os.path.join(path, patient, f'{patient}_flair.nii'))
        raw_imgs.append(img)
        if patient == 'BraTS20_Training_355':
            mask = load_niib_data(os.path.join(path, 'BraTS20_Training_355/W39_1998.09.19_Segm.nii'))
        else:
            mask = load_niib_data(os.path.join(path, patient, f'{patient}_seg.nii'))
        if np.all(mask == 0):
            continue
        mask[mask > 0] = 1

        mask_mask_slice = (mask != 0).any(axis=0).any(axis=0)

        idxs_mask = np.where(mask_mask_slice)[0]

        clip_percentile = (0, np.percentile(img, 95))
        img = np.clip(img, clip_percentile[0], clip_percentile[1])

        img = img[..., idxs_mask]
        mask = mask[..., idxs_mask]

        # scale the images to [0, 1]
        img = scale_images(img, *clip_percentile)

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        processed_imgs.append(img)
        # split in slices and save only slices with tumor 
        os.makedirs(os.path.join(output_path, f'{patient}'), exist_ok=True)
        for i in range(img.shape[2]):
            np.savez(os.path.join(output_path, f'{patient}', f'{patient}_{idxs_mask[i]}'), img=img[:,:,i])
            np.savez(os.path.join(output_path, f'{patient}', f'{patient}_{idxs_mask[i]}_mask'), mask=mask[:,:,i])


print('n_samples', len(processed_imgs))
