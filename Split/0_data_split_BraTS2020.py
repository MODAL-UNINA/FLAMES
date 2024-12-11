# %%
import os
import pickle
import numpy as np
from tqdm import tqdm
from itertools import combinations
if os.getcwd().split('/')[-1] != 'FL4':
    os.chdir('FL4')

def select_patient_combination_with_closest_images(selectable_folds, number_img, dataset_name):

    diff = float('inf')
    best_combination = []
    best_num_images = 0

    max = 4
    for r in tqdm(range(1, max), desc='Selecting patient combination'):
        for combination in combinations(selectable_folds, r):
            total_images = sum(len(os.listdir(folder)) for folder in combination)
            
            if np.abs(total_images - number_img) < diff:
                if total_images == number_img:
                    best_combination = combination
                    best_num_images = total_images
                    break
                diff = np.abs(total_images - number_img)
                best_combination = combination
                best_num_images = total_images
    
    if best_combination:
        for folder in best_combination:
            selectable_folds.remove(folder)
        
        print(f'Selected patient folders: {best_combination} with {best_num_images} images instead of {number_img}')
        return list(best_combination), best_num_images, selectable_folds
    else:
        print("No suitable patient folders found.")
        return [], 0, selectable_folds
    

def split_data(path, dataset_name, split, split_test_clients, number_img_server_train_reduced, number_img_server_test, number_img_server_test_reduced,
                dict_name, output_dir):


    data = os.listdir(path)
    # remove files that are not directories
    data = [d for d in data if os.path.isdir(path + '/' + d)]
    path_folds = [path + '/' + d for d in data]

    n = len(data)
    print('Number of patients:', n)
    total_images = [len(os.listdir(t)) for t in path_folds]    
    selectable_folds = path_folds.copy()


    if number_img_server_train_reduced != 0 and number_img_server_test != 0 and number_img_server_test_reduced !=0:
        selected_patient_train_reduced, num_images_train_reduced, selectable_folds = select_patient_combination_with_closest_images(selectable_folds, number_img_server_train_reduced, dataset_name)
        selected_patient_test_reduced, num_images_test_reduced, selectable_folds = select_patient_combination_with_closest_images(selectable_folds, number_img_server_test_reduced, dataset_name)
        selected_patient_server_test, num_images_server_test, selectable_folds = select_patient_combination_with_closest_images(selectable_folds, number_img_server_test, dataset_name)
    else:
        selected_patient_train_reduced = []
        num_images_train_reduced = 0
        selected_patient_test_reduced = []
        num_images_test_reduced = 0
        selected_patient_server_test = []
        num_images_server_test = 0
    portion = int(n*split)
    train_client = selectable_folds[:portion]
    test_client = selectable_folds[portion:]
    number_img_train_client = [len(os.listdir(t)) for t in train_client]
    number_img_test_client = [len(os.listdir(t)) for t in test_client]
    print(f'Number of images in train client: {np.sum(number_img_train_client)}')
    print(f'Number of images in test client: {np.sum(number_img_test_client)}')
    print(f'Number of images in test server: {num_images_server_test}')
    print(f'Number of images in train server reduced: {num_images_train_reduced}')
    print(f'Number of images in test server reduced: {num_images_test_reduced}')

    assert np.sum(number_img_train_client) + np.sum(number_img_test_client) + num_images_train_reduced + num_images_server_test + num_images_test_reduced == np.sum(total_images), 'Error in splitting data'
    
    dict_data = {
        'train_client': train_client,
        'test_client': test_client,
        'test_server': selected_patient_server_test,
        'train_server_reduced': selected_patient_train_reduced,
        'test_server_reduced': selected_patient_test_reduced
    }
    dict_data_split = {}

    for key in dict_data.keys():
        dict_data_split[key] = {folder.split('/')[-1]: {'images': [folder+'/'+img for img in os.listdir(folder) if not img.endswith('_mask.npz')], 'masks': [folder+'/'+mask for mask in os.listdir(folder) if mask.endswith('_mask.npz')]} for folder in dict_data[key]}
        assert len(dict_data_split[key]) == len(dict_data[key]), 'Error in splitting data'
    print(len(dict_data_split['train_client'].keys()), 'patients')
    
    n_samples = 0
    for pt in dict_data_split['train_client'].keys():
        n_samples += len(dict_data_split['train_client'][pt]['images'])
    print('n_samples', n_samples)
        
    with open(f'{output_dir}/{dict_name}', 'wb') as f:
        pickle.dump(dict_data_split, f)

    return dict_data_split, n_samples

# %%

split = 0.8
split_test_clients = 0.2
output_dir = '../FL4/Split_new_noblack'
dataset_name = 'BraTS2020'
path ='../Datasets/BraTS2020_preprocessed_noblack'
os.makedirs(output_dir, exist_ok=True)
dict_name = f'{dataset_name}_split.pkl'

if dict_name == f'{dataset_name}_split.pkl':
    number_img_server_test = 200  
    number_img_server_train_reduced = 200
    number_img_server_test_reduced = 50

dict_kits, n_samples = split_data(path, dataset_name, split, split_test_clients, number_img_server_train_reduced, number_img_server_test, number_img_server_test_reduced, dict_name, output_dir)

# %%
