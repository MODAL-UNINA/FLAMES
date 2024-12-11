
# %%
import os
import pickle as pkl
from collections import defaultdict

if os.getcwd().split('/')[-1] != 'FL4':
    os.chdir('../FL4')
def group_images_and_masks_by_patient(image_paths, mask_paths):

    patient_data = defaultdict(lambda: {"images": [], "masks": []})
    image_paths.sort()
    mask_paths.sort()
    for img_path, mask_path in zip(image_paths, mask_paths):

        filename = os.path.basename(img_path)
        patient_id = filename.split('_')[0] 
        patient_data[patient_id]["images"].append(img_path)
        patient_data[patient_id]["masks"].append(mask_path)

    print('Total number of patients:', len(patient_data))
    print('Total number of images:', sum(len(patient_data[patient_id]["images"]) for patient_id in patient_data))
    
    return patient_data


def divide_images_and_masks(dataset_name, path, number_img_server_test, number_img_server_train_reduced, 
                            number_img_server_test_reduced, split, split_test_clients,
                            output_dir, dict_name):

    print('Dataset:', dataset_name)
    print('\n')

    test_server_list_total = []
    train_server_reduced_list_total = []
    test_server_reduced_list_total = []
    train_client_list_total = []
    test_client_list_total = []

    
    files_img = os.listdir(path)
    files_img = [f for f in files_img if not f.endswith('.json')]
    
    images_to_delete = pkl.load(open('../Datasets/Breast_US_images_to_delete.pkl', 'rb'))

    for f in files_img:
        if f == 'malignant' or f == 'normal':
            if dict_name == f'{dataset_name}_split.pkl':
                number_img_server_test = 20
                number_img_server_train_reduced = 40
                number_img_server_test_reduced = 15
            elif dict_name == f'{dataset_name}_split_centralized.pkl':
                number_img_server_test = 0
                number_img_server_train_reduced = 0
                number_img_server_test_reduced = 0
        print(f"Folder: {f}")


        mask_paths = [os.path.join(path, f, img) for img in os.listdir(os.path.join(path, f)) if img.endswith('mask.npz') and img.split('.')[0].split('_')[0] not in images_to_delete]
        image_paths = [f.replace('_mask.npz', '.npz') for f in mask_paths]

        assert len(image_paths) == len(mask_paths)

        patient_data = group_images_and_masks_by_patient(image_paths, mask_paths)
        remaining_data = patient_data.copy()
        if number_img_server_test != 0 and number_img_server_train_reduced != 0 and number_img_server_test_reduced != 0:
            test_server_list = list(patient_data.items())[:number_img_server_test]
            train_server_reduced_list = list(patient_data.items())[number_img_server_test:number_img_server_test + number_img_server_train_reduced]
            test_server_reduced_list = list(patient_data.items())[number_img_server_test + number_img_server_train_reduced:number_img_server_test + number_img_server_train_reduced + number_img_server_test_reduced]
            
            print(f"Test server: {len(test_server_list)} images")
            print(f"Train server reduced: {len(train_server_reduced_list)} images")
            print(f"Test server reduced: {len(test_server_reduced_list)} images")

            test_server_list_total.extend(test_server_list)
            train_server_reduced_list_total.extend(train_server_reduced_list)
            test_server_reduced_list_total.extend(test_server_reduced_list)

            remaining_data = {k: v for k, v in list(patient_data.items()) if k not in [t[0] for t in test_server_list] + [t[0] for t in train_server_reduced_list] + [t[0] for t in test_server_reduced_list]}
            print(f"Remaining data: {len(remaining_data)}")
    
        train_client_list = list(remaining_data.items())[:int(len(remaining_data) * split)]
        test_client_list = list(remaining_data.items())[int(len(remaining_data) * split):]
    
        assert len(train_client_list) + len(test_client_list) == len(remaining_data), f'{len(train_client_list)} + {len(test_client_list)} != {len(remaining_data)}'
    
        print(f"Train client: {len(train_client_list)} images")
        print(f"Test client: {len(test_client_list)} images")
        print('\n')
        
        train_client_list_total.extend(train_client_list)
        test_client_list_total.extend(test_client_list)

    train_client = {train_client_list_total[i][0].split('.')[0]: {'images': train_client_list_total[i][1]['images'], 'masks': train_client_list_total[i][1]['masks']} for i in range(len(train_client_list_total))}
    test_client = {test_client_list_total[i][0].split('.')[0]: {'images': test_client_list_total[i][1]['images'], 'masks': test_client_list_total[i][1]['masks']} for i in range(len(test_client_list_total))}
    test_server = {test_server_list_total[i][0].split('.')[0]: {'images': test_server_list_total[i][1]['images'], 'masks': test_server_list_total[i][1]['masks']} for i in range(len(test_server_list_total))}
    train_server_reduced = {train_server_reduced_list_total[i][0].split('.')[0]: {'images': train_server_reduced_list_total[i][1]['images'], 'masks': train_server_reduced_list_total[i][1]['masks']} for i in range(len(train_server_reduced_list_total))}
    test_server_reduced = {test_server_reduced_list_total[i][0].split('.')[0]: {'images': test_server_reduced_list_total[i][1]['images'], 'masks': test_server_reduced_list_total[i][1]['masks']} for i in range(len(test_server_reduced_list_total))}


        
    dict_data = {
        "train_client": train_client,
        "test_client": test_client,
        "test_server": test_server,
        "train_server_reduced": train_server_reduced,
        "test_server_reduced": test_server_reduced
    }

    n_samples = 0
    for pt in dict_data['train_client'].keys():
        n_samples += len(dict_data['train_client'][pt]['images'])
    print('n_samples', n_samples)

    print('current directory:', os.getcwd())

    with open(f'{output_dir}/{dict_name}', 'wb') as f:
        pkl.dump(dict_data, f)

    return n_samples

# %%


number_img_server_test = 0  
number_img_server_train_reduced = 0 
number_img_server_test_reduced = 0

split = 0.8
split_test_clients = 0.2

dataset_name = 'Breast_ultrasound'
path = f'../Datasets/{dataset_name}_preprocessed_noblack'
output_dir = '../FL4/Split_new_noblack'
os.makedirs(output_dir, exist_ok=True)

# SET THE OUTPUT DIRECTORY
dict_name = f'{dataset_name}_split.pkl'

if dict_name == f'{dataset_name}_split.pkl':
    number_img_server_test = 20
    number_img_server_train_reduced = 40
    number_img_server_test_reduced = 15

n_samples = divide_images_and_masks(dataset_name, path, number_img_server_test, number_img_server_train_reduced, 
                                    number_img_server_test_reduced, split, split_test_clients,
                                    output_dir, dict_name)

# %%
