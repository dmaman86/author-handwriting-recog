import json
import os
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf

def save_config(path: str, config: dict) -> None:
    """
    Save the configuration dictionary to a file.

    Args:
        path (str): The file path where the configuration will be saved.
        config (dict): The configuration dictionary to save.
    """
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

def load_config(path: str, config: str) -> dict:
    """
    Load the configuration dictionary from a file.

    Args:
        path (str): The file path where the configuration is saved.
        config (str): The name of the configuration file to load.

    Returns:
        dict: The loaded configuration dictionary.
    """
    with open(os.path.join(path, config), 'r') as f:
        return json.load(f)
    
def load_pickle(path: str) -> object:
    """
    Load a pickle file from the specified path.

    Args:
        path (str): The file path to load the pickle from.

    Returns:
        object: The loaded object from the pickle file.
    """

    with open(path, 'rb') as f:
        return pickle.load(f)
    
def extract_pkl_content_to_npy_folder(content: dict, output_folder: str) -> None:
    """
    Extract content from a dictionary and save it as .npy files in the specified folder.

    Args:
        content (dict): The dictionary containing data to be saved.
        output_folder (str): The folder where the .npy files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename, array in tqdm(content.items(), desc="Extracting to .npy"):
        output_path = os.path.join(output_folder, filename)
        np.save(output_path, array)

def load_npy_patch(patch_path: str, resize_shape: tuple[int, int, int]) -> np.ndarray:
  patch = np.load(patch_path)  # Load .npy file as (H, W) or (H, W, 1)

  if patch.ndim == 2:
    patch = patch[..., np.newaxis]  # Convert (H, W) to (H, W, 1)

  patch = tf.image.resize(patch, resize_shape[:2])  # Resize to match model input shape
  patch = patch / 255.0  # Normalize to [0, 1]
  return np.expand_dims(patch, axis=0)  # Add batch dimension: (1, H, W, C)

def get_patches(patch_ids: list[str], binary_dir: str, resize_shape: tuple[int, int] = (60, 53)) -> list[np.ndarray]:
  '''
  Loads and resizes multiple image patches from a given directory.
  Args:
    patch_ids (list[str]): List of .npy filenames to load.
    binary_dir (str): Directory containing the .npy patch files.
    resize_shape (tuple[int, int]): Target size (height, width) for resizing.
  Returns:
    list[np.ndarray]: List of loaded and resized image patches.
  '''
  patches: list[np.ndarray] = []

  for patch_id in patch_ids:
    patch = np.load(os.path.join(binary_dir, patch_id))
    patch = tf.image.resize(patch[..., np.newaxis], resize_shape)
    patch = patch.numpy() / 255.0
    patches.append(patch)

  return patches