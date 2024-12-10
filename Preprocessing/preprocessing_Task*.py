# %%
import os
import cv2
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from utils import load_niib_data, scale_images

fixed_shape = (240, 240)
batch_size = 10

# MRI
preprocess_task01 = False
# CT
preprocess_task06 = False
preprocess_task07 = True

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


        if dataset == "Task01_BrainTumour":
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

        elif dataset == "Task06_Lung" or dataset == "Task07_Pancreas":
            if dataset == "Task07_Pancreas":
                mask[mask == 1] = 0
                mask[mask == 2] = 1
            if img.shape[:2] != fixed_shape:
                assert img.shape[0] == img.shape[1]
                scale = fixed_shape[0] / img.shape[0]

                img = np.stack(
                    [
                        cv2.resize(
                            img[..., i],
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                        for i in range(img.shape[-1])
                    ],
                    axis=-1,
                )
                mask = np.stack(
                    [
                        cv2.resize(
                            mask[..., i],
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for i in range(img.shape[-1])
                    ],
                    axis=-1,
                )

            mask_mask_slice = (mask != 0).any(axis=0).any(axis=0)

            idxs_mask = np.where(mask_mask_slice)[0]
            idxs_mask = idxs_mask[np.abs(np.mean(idxs_mask) - idxs_mask) <= 20]

            clip_percentile = (-1000, np.percentile(img, 95))
            clip_range = (-1000, 1000)
            img = np.clip(img, clip_percentile[0], clip_percentile[1])

            img = img[..., idxs_mask]
            mask = mask[..., idxs_mask]

            img = np.clip(img, clip_range[0], clip_range[1])
            img = scale_images(img, *clip_range)

            start_range = 0
            end_range = img.shape[2]
            for i in range(start_range, end_range):
                img_slice = img[:, :, i]
                mask_slice = mask[:, :, i]
                processed_images.append(img_slice)
                assert (
                    img_slice.shape == fixed_shape
                ), f"Image shape {img_slice.shape} different from {fixed_shape}"
                # save
                if not os.path.exists(f"{output_path}/imagesTr"):
                    os.makedirs(f"{output_path}/imagesTr")
                if not os.path.exists(f"{output_path}/labelsTr"):
                    os.makedirs(f"{output_path}/labelsTr")

                img_slice = img_slice.astype(np.float32)
                mask_slice = mask_slice.astype(np.float32)
                np.savez(
                    f"{output_path}/imagesTr/{case_id}_{idxs_mask[i]}", img=img_slice
                )
                np.savez(
                    f"{output_path}/labelsTr/{case_id}_{idxs_mask[i]}", mask=mask_slice
                )

    print("Total images:", len(raw_images))
    print("Total images:", len(processed_images))

    return len(processed_images)


# %%
if preprocess_task01:
    dataset = "Task01_BrainTumour"
elif preprocess_task06:
    dataset = "Task06_Lung"
elif preprocess_task07:
    dataset = "Task07_Pancreas"

path = f"../Datasets/{dataset}"
output_path = f"../Datasets/{dataset}_preprocessed_noblack"
if dataset == "Task07_Pancreas":
    path = path + f"/{dataset}"


folders = os.listdir(path)
folders = [f for f in folders if os.path.isdir(os.path.join(path, f))]
os.makedirs(output_path, exist_ok=True)

n_samples = preprocess_images(path, folders, fixed_shape, dataset)
print('n_samples', n_samples)

