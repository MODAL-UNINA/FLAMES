import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


def analyze_image_values(images, title):
    """
    Funzione che calcola e visualizza un istogramma della distribuzione dei valori di pixel per un set di immagini.
    """
    all_pixels = np.concatenate([image.flatten() for image in images], axis=0)

    plt.figure(figsize=(10, 6))
    plt.hist(all_pixels, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.yscale('log')
    if 'Processed' in title:
        plt.xlim(0, 1)
    plt.savefig(f'{title}.png')
    plt.show()

def load_niib_data(file_path, dataset_name=None):
    nib_imag = nib.load(file_path)
    x, y, z = nib.aff2axcodes(nib_imag.affine)
    img = nib_imag.get_fdata()

    if dataset_name == 'kits23':
        if img.shape[1:] != (512,512):
            print(f"Image {file_path} shape: {img.shape}")
    if x != "R":
        img = np.flip(img, axis=0)
    if y != "A":
        img = np.flip(img, axis=1)
    if z != "S":
        img = np.flip(img, axis=2)

    if dataset_name == 'kits23':
        img = np.rot90(img, 2, axes=(1, 2))
    else:
        img = np.rot90(img, axes=(0, 1))
    if dataset_name == 'kits23':
        if img.shape[1:] != (512,512):
            print(f"Image {file_path} shape: {img.shape}")

    return img

def scale_images(image, min_val, max_val):
    """
    Funzione che scala i valori di pixel delle slices di un set di slices usando i valori nel 95esimo percentile dell'immagine 3D
    """
    image = (image - min_val) / (max_val - min_val)

    return image