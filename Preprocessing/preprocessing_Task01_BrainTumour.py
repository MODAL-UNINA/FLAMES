# %%
import os
import cv2
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from utils import load_niib_data, scale_images

fixed_shape = (240, 240)
batch_size = 10


# %%

def preprocess_images(path, folders, fixed_shape, dataset):
    assert fixed_shape[0] == fixed_shape[1]
    raw_images = []
    processed_images = []

    folder_path = os.path.join(path, "imagesTr")
    for file in tqdm(
        os.listdir(folder_path),
        total=len(os.listdir(folder_path)),
        desc="Processing images",
    ):
        if file.startswith("."):
            continue
        case_id = file.split(".")[0]

        img = load_niib_data(os.path.join(folder_path, file))
        raw_images.append(img)

        mask = load_niib_data(os.path.join(path, "labelsTr", file))

        if np.all(mask == 0):
            continue

        mask[mask > 0] = 1
        for channel in [0]:
            img = img[:, :, :, channel]

            mask_mask_slice = (mask != 0).any(axis=0).any(axis=0)

            idxs_mask = np.where(mask_mask_slice)[0]
            idxs_mask = idxs_mask[np.abs(np.mean(idxs_mask) - idxs_mask) <= 20]

            clip_percentile = (0, np.percentile(img, 95))
            img = np.clip(img, clip_percentile[0], clip_percentile[1])

            img = img[..., idxs_mask]
            mask = mask[..., idxs_mask]

            img = scale_images(img, *clip_percentile)

            plt.imshow(img[:, :, img.shape[2] // 2], cmap="gray")

            start_range = 0
            end_range = img.shape[2]
            for i in range(start_range, end_range):
                mask_slice = mask[:, :, i]
                if np.all(mask_slice == 0):
                    continue
                img_slice = img[:, :, i]
                processed_images.append(img_slice)
                assert (
                    img_slice.shape == fixed_shape
                ), f"Image shape {img_slice.shape} different from {fixed_shape}"
                # save
                if not os.path.exists(f"{output_path}/imagesTr"):
                    os.makedirs(f"{output_path}/imagesTr")
                if not os.path.exists(f"{output_path}/labelsTr"):
                    os.makedirs(f"{output_path}/labelsTr")
                np.savez(
                    f"{output_path}/imagesTr/{case_id}_{idxs_mask[i]}",
                    img=img_slice,
                )
                np.savez(
                    f"{output_path}/labelsTr/{case_id}_{idxs_mask[i]}",
                    mask=mask_slice,
                )

    print("Total images:", len(raw_images))
    print("Total images:", len(processed_images))

    return len(processed_images)


# %%
dataset = "Task01_BrainTumour"

path = f"../Datasets/{dataset}"
output_path = f"../Datasets/{dataset}_preprocessed_noblack"


folders = os.listdir(path)
folders = [f for f in folders if os.path.isdir(os.path.join(path, f))]
os.makedirs(output_path, exist_ok=True)

n_samples = preprocess_images(path, folders, fixed_shape, dataset)
print('n_samples', n_samples)

