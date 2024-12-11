import os
import cv2
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

class CustomDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        augmentations=None,
        transform=None,
        mask_transform=None,
        batch_size=10,
        original_data = False
    ):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations
        self.transform = transform
        self.mask_transform = mask_transform
        self.batch_size = batch_size
        self.original_data = original_data
        self.seed = 42
        self.round = 0
        self.epoch = 0

        print(
            "self.augmentations",
            len(self.augmentations) if self.augmentations is not None else None,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch
        print("SETTING dataset EPOCH: ", epoch)

    def set_round(self, round):
        self.round = round
        print("SETTING dataset ROUND: ", round)

    def __len__(self):
        if self.original_data:
            return len(self.images) * 2
        return len(self.images)

    def __getitem__(self, idx):
        idx_real = idx
        if self.original_data:
            idx_real = idx % len(self.images)

        image = self.images[idx_real]
        mask = self.masks[idx_real]

        if self.augmentations is not None and idx_real != idx:
            transformation = self.augmentations[f"({self.round}, {self.epoch})"][
                idx_real
                ]

            image, mask = self.apply_augmentation(image, mask, transformation)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    @staticmethod
    def can_print(i, r):
        return i == 4 and r == 0

    def calculate_class_distribution(self, dataloader):
        class_counts = {0: 0, 1: 0}
        total_pixels = 0

        for _, masks in dataloader:
            masks = masks.float().view(-1)
            class_counts[0] += (masks == 0).sum().item()
            class_counts[1] += (masks == 1).sum().item()
            total_pixels += masks.numel()

        class_0_percentage = class_counts[0] / total_pixels * 100
        class_1_percentage = class_counts[1] / total_pixels * 100

        print(f"Classe 0 (background): {class_0_percentage:.2f}%")
        print(f"Classe 1 (foreground): {class_1_percentage:.2f}%")


    def histogram_images_plot(self, images, masks, dataset, set_type):

        images = np.array(images)
        masks = np.array(masks)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].hist(images.flatten(), bins=50, color="b")
        axs[0, 0].set_title("Images histogram")
        axs[0, 1].hist(masks.flatten(), bins=50, color="r")
        axs[0, 1].set_title("Masks histogram")
        axs[1, 0].imshow(images[0], cmap="gray")
        axs[1, 0].set_title("Image")
        axs[1, 1].imshow(masks[0], cmap="gray")
        axs[1, 1].set_title("Mask")
        plt.suptitle(f"{set_type} {dataset} dataset")
        plt.savefig(f"histogram_{dataset}_{set_type}.png")

    def apply_flip(self, img, mask, config):
        which_flip = config["which"]
        apply_flip = config["apply"]
        if apply_flip:
            img = cv2.flip(img, which_flip)
            mask = cv2.flip(mask, which_flip)
        return img, mask

    def apply_rotate(self, img, mask, config):
        angle = config["angle"]
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        img = cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4
        )
        mask = cv2.warpAffine(
            mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST
        )
        return img, mask

    def apply_zoom(self, img, mask, config):
        zoom = config["scale"]
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 0, zoom)
        img = cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4
        )
        mask = cv2.warpAffine(
            mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST
        )
        return img, mask

    def apply_shift(self, img, mask, config):
        dx = config["dx"]
        dy = config["dy"]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4
        )
        mask = cv2.warpAffine(
            mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST
        )
        return img, mask

    def apply_stretch(self, img, mask, config):
        scale_x = config["scale_x"]
        scale_y = config["scale_y"]
        M = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
        img = cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4
        )
        mask = cv2.warpAffine(
            mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST
        )
        return img, mask

    def apply_augmentation(self, img, mask, transformation):
        flip = transformation["flip"]
        rotate = transformation["rotate"]
        zoom = transformation["zoom"]
        shift = transformation["shift"]
        stretch = transformation["stretch"]

        # apply flip
        img, mask = self.apply_flip(img, mask, flip)

        # apply rotate
        img, mask = self.apply_rotate(img, mask, rotate)

        # apply zoom
        img, mask = self.apply_zoom(img, mask, zoom)

        # apply shift
        img, mask = self.apply_shift(img, mask, shift)

        # apply stretch
        img, mask = self.apply_stretch(img, mask, stretch)

        # clip to 0-1
        img = np.clip(img, 0, 1)

        return img, mask

    def load_data_client(
        self,
        dataset,
        data_augmentation: bool = False,
        debug=False,
        return_train_eval_loader=False,
        original_data=False,
    ):
        """Load dictionary with paths to images and masks"""
        print("current directory:", os.getcwd())
        data_dict_path = f"../Datasets/Split_new_noblack/{dataset}_split.pkl"

        with open(data_dict_path, "rb") as f:
            dataset_dict = pkl.load(f)

        path_train_client = dataset_dict["train_client"]
        path_test_client = dataset_dict["test_client"]

        imgs = [
            np.load(f"{id}")["img"].astype(np.float32)
            for k in range(len(path_train_client.keys()))
            for id in path_train_client[list(path_train_client.keys())[k]]["images"]
            if os.path.exists(id)
        ]
        masks = [
            np.load(f"{id}")["mask"].astype(np.float32)
            for k in range(len(path_train_client.keys()))
            for id in path_train_client[list(path_train_client.keys())[k]]["masks"]
            if os.path.exists(id)
        ]

        assert all(set(np.unique(m)).issubset({0, 1}) for m in masks)

        print("Num images:", len(imgs))
        assert len(imgs) == len(masks)

        if data_augmentation:
            with open(
                f"../Datasets/{dataset}_preprocessed_noblack/augmentations.json", "r"
            ) as f:
                augmentations = json.load(f)

            online_mode = isinstance(augmentations, dict)
            if not online_mode:
                dataset_orig = dataset.split("__")[0]
                if dataset_orig == "Breast_ultrasound":
                    num_augmentation = 10
                elif dataset_orig == "OTU_2d":
                    num_augmentation = 10
                elif dataset_orig == "Task06_Lung":
                    num_augmentation = 10
                elif dataset_orig == "Task07_Pancreas":
                    num_augmentation = 2
                elif dataset_orig == "kits23":
                    num_augmentation = 5
                elif dataset_orig == "BraTS2020":
                    num_augmentation = 10
                elif dataset_orig == "Task01_BrainTumour":
                    num_augmentation = 1

                augmentations = augmentations[:num_augmentation]

                imgs_augms = []
                masks_augms = []

                print("Data augmentation..")
                for i in range(len(imgs)):
                    img = imgs[i]
                    mask = masks[i]
                    assert np.unique(mask).tolist() == [
                        0,
                        1,
                    ], f"Unique values in mask: {np.unique(mask)}"
                    assert img.shape == mask.shape

                    for transformation in augmentations:

                        # apply augmentation
                        img_aug, mask_aug = self.apply_augmentation(
                            img, mask, transformation
                        )
                        assert set(np.unique(mask_aug)).issubset(
                            {0, 1}
                        ), f"Unique values in mask: {np.unique(mask_aug)}"
                        assert img_aug.dtype == np.float32
                        assert mask_aug.dtype == np.float32
                        imgs_augms.append(img_aug)
                        masks_augms.append(mask_aug)

                imgs.extend(imgs_augms)
                masks.extend(masks_augms)
            else:
                assert len(augmentations["(0, 0)"]) == len(imgs)

        imgs_test = [
            np.load(f"{id}")["img"].astype(np.float32)
            for k in range(len(path_test_client.keys()))
            for id in path_test_client[list(path_test_client.keys())[k]]["images"]
        ]
        masks_test = [
            np.load(f"{id}")["mask"].astype(np.float32)
            for k in range(len(path_test_client.keys()))
            for id in path_test_client[list(path_test_client.keys())[k]]["masks"]
        ]

        assert all(set(np.unique(m)).issubset({0, 1}) for m in masks_test)

        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        mask_transform = ToTensor()

        dataset_train_client = CustomDataset(
            imgs,
            masks,
            augmentations=augmentations if data_augmentation and online_mode else None,
            transform=transform,
            mask_transform=mask_transform,
            original_data=original_data
        )
        dataset_test_client = CustomDataset(
            imgs_test, masks_test, transform=transform, mask_transform=mask_transform
        )
        trainloader_client = DataLoader(
            dataset_train_client, batch_size=self.batch_size, shuffle=True
        )
        testloader_client = DataLoader(
            dataset_test_client, batch_size=self.batch_size, shuffle=False
        )

        for img, mask in trainloader_client:
            assert set(np.unique(mask)).issubset({0, 1})
        for img, mask in testloader_client:
            assert set(np.unique(mask)).issubset({0, 1})

        print("Loaded images and masks of dataset ", dataset)
        print("Number of train patients:", len(path_train_client))
        print("Number of test patients:", len(path_test_client))

        if return_train_eval_loader:
            dataset_train_eval_client = CustomDataset(
                imgs, masks, transform=transform, mask_transform=mask_transform
            )
            trainloader_eval_client = DataLoader(
                dataset_train_eval_client, batch_size=self.batch_size, shuffle=False
            )
            return trainloader_client, testloader_client, trainloader_eval_client

        return trainloader_client, testloader_client

    def load_data_server(
        self,
        datasets,
        debug: bool = False,
        apply_augmentations: bool = False,
        return_train_eval_loader: bool = False,
        original_data=False,
    ):
        """Load dictionary with paths to images and masks"""
        imgs_test_total = []
        masks_test_total = []
        imgs_train_red_total = []
        masks_train_red_total = []
        imgs_test_red_total = []
        masks_test_red_total = []
        for dataset in datasets:
            dataset_split = dataset.split("__")
            if len(dataset_split) == 2:
                if dataset_split[1] != "0":
                    continue
                dataset = dataset_split[0]

            data_dict_path = f"../Datasets/Split_new_noblack/{dataset}_split.pkl"

            with open(data_dict_path, "rb") as f:
                dataset_dict = pkl.load(f)

            if apply_augmentations:
                with open(
                    f"../Datasets/{dataset}_preprocessed_noblack/augmentations.json",
                    "r",
                ) as f:
                    augmentations = json.load(f)
            else:
                augmentations = None

            path_test_server = dataset_dict["test_server"]
            path_train_server_reduced = dataset_dict["train_server_reduced"]
            path_test_server_reduced = dataset_dict["test_server_reduced"]

            imgs_test = [
                np.load(f"{id}")["img"]
                for k in range(len(path_test_server.keys()))
                for id in path_test_server[list(path_test_server.keys())[k]]["images"]
            ]
            masks_test = [
                np.load(f"{id}")["mask"]
                for k in range(len(path_test_server.keys()))
                for id in path_test_server[list(path_test_server.keys())[k]]["masks"]
            ]

            imgs_train_red = [
                np.load(f"{id}")["img"]
                for k in range(len(path_train_server_reduced.keys()))
                for id in path_train_server_reduced[
                    list(path_train_server_reduced.keys())[k]
                ]["images"]
            ]
            masks_train_red = [
                np.load(f"{id}")["mask"]
                for k in range(len(path_train_server_reduced.keys()))
                for id in path_train_server_reduced[
                    list(path_train_server_reduced.keys())[k]
                ]["masks"]
            ]

            imgs_test_red = [
                np.load(f"{id}")["img"]
                for k in range(len(path_test_server_reduced.keys()))
                for id in path_test_server_reduced[
                    list(path_test_server_reduced.keys())[k]
                ]["images"]
            ]
            masks_test_red = [
                np.load(f"{id}")["mask"]
                for k in range(len(path_test_server_reduced.keys()))
                for id in path_test_server_reduced[
                    list(path_test_server_reduced.keys())[k]
                ]["masks"]
            ]

            imgs_test_total.extend(imgs_test)
            masks_test_total.extend(masks_test)
            imgs_train_red_total.extend(imgs_train_red)
            masks_train_red_total.extend(masks_train_red)
            imgs_test_red_total.extend(imgs_test_red)
            masks_test_red_total.extend(masks_test_red)

            print("Loaded images and masks of dataset ", dataset)
            print("Number of train patients:", len(path_train_server_reduced))
            print("Number of test patients:", len(path_test_server))
            print("Number of test patients reduced:", len(path_test_server_reduced))
            print("Num images:", len(imgs_test))
            print("Num images train reduced:", len(imgs_train_red))
            print("Num images test reduced:", len(imgs_test_red))

        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        mask_transform = ToTensor()

        dataset_test_server = CustomDataset(
            imgs_test_total,
            masks_test_total,
            transform=transform,
            mask_transform=mask_transform,
        )

        dataset_train_server_reduced = CustomDataset(
            imgs_train_red_total,
            masks_train_red_total,
            augmentations=augmentations,
            transform=transform,
            mask_transform=mask_transform,
            original_data=original_data
        )

        dataset_test_server_reduced = CustomDataset(
            imgs_test_red_total,
            masks_test_red_total,
            transform=transform,
            mask_transform=mask_transform,
        )

        testloader_test_server = DataLoader(
            dataset_test_server, batch_size=self.batch_size, shuffle=True
        )
        testloader_train_server_reduced = DataLoader(
            dataset_train_server_reduced, batch_size=self.batch_size, shuffle=True
        )
        testloader_test_server_reduced = DataLoader(
            dataset_test_server_reduced, batch_size=self.batch_size, shuffle=True
        )

        if return_train_eval_loader:
            dataset_train_eval_client = CustomDataset(
                imgs_train_red_total,
                masks_train_red_total,
                transform=transform,
                mask_transform=mask_transform,
            )
            trainloader_eval_client = DataLoader(
                dataset_train_eval_client, batch_size=self.batch_size, shuffle=False
            )
            return (
                testloader_test_server,
                testloader_train_server_reduced,
                testloader_test_server_reduced,
                trainloader_eval_client,
            )

        return (
            testloader_test_server,
            testloader_train_server_reduced,
            testloader_test_server_reduced,
        )
