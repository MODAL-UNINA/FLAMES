# %%
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

def group_images_and_masks_by_patient(image_paths, mask_paths):

    patient_data = defaultdict(lambda: {"images": [], "masks": []})
    
    for img_path, mask_path in zip(image_paths, mask_paths):

        filename = os.path.basename(img_path)
        patient_id = filename.split('_')[1]
        patient_data[patient_id]["images"].append(img_path)
        patient_data[patient_id]["masks"].append(mask_path)

    print('total number of patients', len(patient_data))
    print('total number of images', sum(len(patient_data[patient_id]["images"]) for patient_id in patient_data))
    
    return patient_data

def select_patient_combination_with_closest_images_and_masks(patient_data, number_img):

    selectable_folds = list(patient_data.keys())
    diff = float('inf')
    best_combination = []
    best_num_images = 0
    
    max = 4
    for r in tqdm(range(1, max), desc="Selecting patient combination"):
        for combination in combinations(selectable_folds, r):
            total_images = sum(len(patient_data[patient_id]["images"]) for patient_id in combination)

            if np.abs(total_images - number_img) < diff:
                if total_images == number_img:
                    best_combination = combination
                    best_num_images = total_images
                    break
                diff = np.abs(total_images - number_img)
                best_combination = combination
                best_num_images = total_images
    print(f"Best combination: {best_combination} ({best_num_images} images), instead of {number_img} images")
    
    remaining_patients = {k: v for k, v in patient_data.items() if k not in best_combination}
    selected_patients = {k: v for k, v in patient_data.items() if k in best_combination}
    
    return list(best_combination), best_num_images, remaining_patients, selected_patients

def split_for_client(remaining_data, split):

    train_client = {}
    test_client = {}
    
    n = len(remaining_data)
    n_train = int(n * split)
    patients = list(remaining_data.keys())
    np.random.shuffle(patients)
    train_patients = patients[:n_train]
    test_patients = patients[n_train:]

    num_imgs_train = 0
    num_imgs_test = 0
    for patient in train_patients:
        train_client[patient] = remaining_data[patient]
        num_imgs_train += len(remaining_data[patient]["images"])

    
    for patient in test_patients:
        test_client[patient] = remaining_data[patient]
        num_imgs_test += len(remaining_data[patient]["images"])
    
    return train_client, test_client, num_imgs_train, num_imgs_test

def split_in_n_clients(client_data, n_clients=3):
    patient_ids = list(client_data.keys())
    split_size = len(patient_ids) // n_clients
    remainder = len(patient_ids) % n_clients
    split_clients = []
    start = 0
    for i in range(n_clients):
        end = start + split_size + (1 if i < remainder else 0)
        split_ids = patient_ids[start:end]
        split_clients.append({pid: client_data[pid] for pid in split_ids})
        start = end
    
    return split_clients

def divide_images_and_masks(dataset_name, path, number_img_server_test, split, split_test_clients, dict_name, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    print('Dataset:', dataset_name)

    files_img = os.listdir(os.path.join(path, 'imagesTr'))
    files_mask = os.listdir(os.path.join(path, 'labelsTr'))
    print(f"Number of images: {len(files_img)}")
    files_img.sort()
    files_mask.sort()
            
    patients_img = {}
    patients_mask = {}

    for file in files_img:
        patient = file.split('_')[1]
        if patient not in patients_img:
            patients_img[patient] = []
        patients_img[patient].append(file)


    for file in files_mask:
        patient = file.split('_')[1]
        if patient not in patients_mask:
            patients_mask[patient] = []
        patients_mask[patient].append(file)

    patients_list = list(patients_img.keys())
    print(f"Total number of patients: {len(patients_list)}")

    patients_imgs = [patients_img[patient] for patient in patients_list]
    patients_masks = [patients_mask[patient] for patient in patients_list]
    
    image_paths = [os.path.join(path, 'imagesTr', img) for sublist in patients_imgs for img in sublist]
    mask_paths = [os.path.join(path, 'labelsTr', mask) for sublist in patients_masks for mask in sublist]
    assert len(image_paths) == len(mask_paths)


    patient_data = group_images_and_masks_by_patient(image_paths, mask_paths)
    if number_img_server_test != 0:
        test_server_patients, test_server_images, patient_data, selected_patients_test_server = select_patient_combination_with_closest_images_and_masks(patient_data, number_img_server_test)

    train_client, test_client, num_imgs_train, num_imgs_test = split_for_client(patient_data, split)
    if number_img_server_test != 0 :
        print(f"Test server: {test_server_images} images")

    print(f"Train client: {len(train_client)} patients with {num_imgs_train} images")
    print(f"Test client: {len(test_client)} patients with {num_imgs_test} images")
    print('\n')
    
    if number_img_server_test!= 0:
        dict_data= {
            "train_client": train_client,
            "test_client": test_client,
            "test_server": selected_patients_test_server,
        }

        train_list = split_in_n_clients(train_client)
        test_list = split_in_n_clients(test_client)

        dict_data_1 = {
            "train_client": train_list[0],
            "test_client": test_list[0],
        }
        dict_data_2 = {
            "train_client": train_list[1],
            "test_client": test_list[1],
        }
        dict_data_3 = {
            "train_client": train_list[2],
            "test_client": test_list[2],
        }
        
        tot_images_train1 = np.sum([len(train_list[0][k]['images']) for k in train_list[0]])
        tot_images_train2 = np.sum([len(train_list[1][k]['images']) for k in train_list[1]])
        tot_images_train3 = np.sum([len(train_list[2][k]['images']) for k in train_list[2]])
        
        print('Lunghezze dei client train e test:', len(train_list[0]), len(train_list[1]), len(train_list[2]),len(test_list[0]), len(test_list[1]), len(test_list[2]))
        dict_data_list = [dict_data_1, dict_data_2, dict_data_3]

        names = [dict_name.split('_')[0] + '_'+ dict_name.split('_')[1] + f'__{n}' + '_split.pkl' for n in range(3)]
        for i, (dict_, name) in enumerate(zip(dict_data_list, names)):
            with open(f'{output_dir}/{name}', 'wb') as f:
                pickle.dump(dict_, f)

        print(f"Salvati {len(dict_data_list)} dizionari nei file: {', '.join(names)}")


    print('n_samples', num_imgs_train)
    with open(f'{output_dir}/{dict_name}', 'wb') as f:
        pickle.dump(dict_data, f)
    print(f"Salvato il dizionario nel file: {dict_name}")

    return num_imgs_train, tot_images_train1, tot_images_train2, tot_images_train3


split = 0.8
split_test_clients = 0.2

datasets = 'Task01_BrainTumour'
output_dir = '../FL4/Split_new_noblack'

for dataset_name in datasets:
    print(f"Dataset: {dataset_name}")

    dict_name = f'{dataset_name}_split.pkl'

    number_img_server_test = 230  

    path = f'../Datasets/{dataset_name}_preprocessed_noblack'
    n_samples, tot_images_train1, tot_images_train2, tot_images_train3 = divide_images_and_masks(dataset_name, path, number_img_server_test, split, split_test_clients, dict_name, output_dir)

# %%