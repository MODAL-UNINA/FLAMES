# %%
import os
import pickle
def build_paths(files, subfolder):
    return [os.path.join(path, subfolder, f) for f in files]

def divide_images_and_masks_otu_2d(dataset_name, output_dir, path, number_img_server_test, number_img_server_train_reduced, number_img_server_test_reduced, split, split_test_clients, dict_name):

    os.makedirs(output_dir, exist_ok=True)
    print('Dataset:', dataset_name)
    print('Path:', path)

    files_images = os.listdir(os.path.join(path, 'images'))
    files_masks = os.listdir(os.path.join(path, 'annotations'))

    files_images.sort()
    files_masks.sort()
    assert len(files_images) == len(files_masks), f'{len(files_images)} != {len(files_masks)}'
    
    n = len(files_images)
    print(f'Total images: {n}')

    test_server_reduced = build_paths(files_images[:number_img_server_test_reduced], 'images')
    test_server_reduced_masks = build_paths(files_masks[:number_img_server_test_reduced], 'annotations')

    test_server = build_paths(files_images[number_img_server_test_reduced:number_img_server_test_reduced + number_img_server_test], 'images')
    test_server_masks = build_paths(files_masks[number_img_server_test_reduced:number_img_server_test_reduced + number_img_server_test], 'annotations')

    train_server_reduced = build_paths(files_images[number_img_server_test_reduced + number_img_server_test:number_img_server_test_reduced + number_img_server_test + number_img_server_train_reduced], 'images')
    train_server_reduced_masks = build_paths(files_masks[number_img_server_test_reduced + number_img_server_test:number_img_server_test_reduced + number_img_server_test + number_img_server_train_reduced], 'annotations')

    remaining_images = files_images[number_img_server_test_reduced + number_img_server_test + number_img_server_train_reduced:]
    remaining_masks = files_masks[number_img_server_test_reduced + number_img_server_test + number_img_server_train_reduced:]

    remaining_n = len(remaining_images)
    print(f'Remaining images: {remaining_n}')

    train_client_n = int(remaining_n * split)

    train_client = build_paths(remaining_images[:train_client_n], 'images')
    train_client_masks = build_paths(remaining_masks[:train_client_n], 'annotations')

    test_client = build_paths(remaining_images[train_client_n:], 'images')
    test_client_masks = build_paths(remaining_masks[train_client_n:], 'annotations')

    total_images = len(train_client) + len(test_client) + len(train_server_reduced) + len(test_server) + len(test_server_reduced)
    total_masks = len(train_client_masks) + len(test_client_masks) + len(train_server_reduced_masks) + len(test_server_masks) + len(test_server_reduced_masks)
    assert total_images == n, f'Total images {total_images} != {n}'
    assert total_masks == n, f'Total masks {total_masks} != {n}'

    print(f'Train client: {len(train_client)} images')
    print(f'Test client: {len(test_client)} images')
    print(f'Train server reduced: {len(train_server_reduced)} images')
    print(f'Test server: {len(test_server)} images')
    print(f'Test server reduced: {len(test_server_reduced)} images')
    print('train_client[0]:', train_client[0])
    
    train_client
    
    split_dict = {
        'train_client': {train_client[i].split('/')[-1]: {'images': [train_client[i]], 'masks': [train_client_masks[i]]} for i in range(len(train_client))},
        'test_client': {test_client[i].split('/')[-1]: {'images': [test_client[i]], 'masks': [test_client_masks[i]]} for i in range(len(test_client))},
        'test_server': {test_server[i].split('/')[-1]: {'images': [test_server[i]], 'masks': [test_server_masks[i]]} for i in range(len(test_server))},
        'train_server_reduced': {train_server_reduced[i].split('/')[-1]: {'images': [train_server_reduced[i]], 'masks': [train_server_reduced_masks[i]]} for i in range(len(train_server_reduced))},
        'test_server_reduced': {test_server_reduced[i].split('/')[-1]: {'images': [test_server_reduced[i]], 'masks': [test_server_reduced_masks[i]]} for i in range(len(test_server_reduced))}
    }

    n_samples = 0
    for pt in split_dict['train_client'].keys():
        n_samples += len(split_dict['train_client'][pt]['images'])
    print('n_samples', n_samples)

    with open(f'{output_dir}/{dict_name}', 'wb') as f:
        pickle.dump(split_dict, f)

    return split_dict, n_samples

# %%

number_img_server_test = 0
number_img_server_train_reduced = 0
number_img_server_test_reduced = 0
split = 0.8
split_test_clients = 0.2

# Esegui la funzione per OTU_2d
dataset_name = 'OTU_2d'
path = f'../Datasets/{dataset_name}_preprocessed_noblack'
output_dir = '../FL4/Split_new_noblack'

dict_name = f'{dataset_name}_split.pkl'

if dict_name == f'{dataset_name}_split.pkl':
    number_img_server_test = 80
    number_img_server_train_reduced = 100
    number_img_server_test_reduced = 50
split_data, n_samples = divide_images_and_masks_otu_2d(dataset_name, output_dir, path, number_img_server_test, number_img_server_train_reduced, number_img_server_test_reduced, split, split_test_clients, dict_name)

# %%
