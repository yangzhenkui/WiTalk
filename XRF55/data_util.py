import os
import logging
import random
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import Dataset
import pynvml

class NpyDataset(Dataset):
    """Dataset for loading .npy files with optional person ID filtering."""
    
    def __init__(self, data_dir, include_person_ids=None, exclude_person_ids=None, transform=None):
        """
        Initializes the dataset.

        Args:
            data_dir (str): Directory containing .npy files.
            include_person_ids (list, optional): List of person IDs to include.
            exclude_person_ids (list, optional): List of person IDs to exclude.
            transform (callable, optional): Optional transform function.

        Raises:
            ValueError: If no valid .npy files are found after filtering.
        """
        self.data_dir = data_dir
        self.transform = transform

        # List all .npy files in the directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

        # Filter by person IDs
        if include_person_ids:
            include_person_ids = [str(pid) for pid in include_person_ids]
            self.file_list = [f for f in self.file_list if f.split("_")[0] in include_person_ids]
        if exclude_person_ids:
            exclude_person_ids = [str(pid) for pid in exclude_person_ids]
            self.file_list = [f for f in self.file_list if f.split("_")[0] not in exclude_person_ids]

        # Validate file list
        if not self.file_list:
            msg = "No .npy files found"
            if include_person_ids:
                msg += f" for include_person_ids: {include_person_ids}"
            elif exclude_person_ids:
                msg += f"; all files excluded by exclude_person_ids: {exclude_person_ids}"
            else:
                msg += " in the specified directory."
            raise ValueError(msg)

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input_data, label) where input_data is a float tensor and label is a long tensor.
        """
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        # Load and convert data
        input_data = np.load(file_path)
        input_data = torch.tensor(input_data, dtype=torch.float32)

        # Extract and convert label (subtract 1 for 0-based indexing)
        label = int(file_name.split('_')[1]) - 1
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            input_data = self.transform(input_data)

        return input_data, label

def check_dir(path):
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def get_logs_model_dir():
    """
    Creates and returns log and model directories based on current date.

    Returns:
        tuple: (log_dir, model_dir) paths.
    """
    current_time = datetime.now().strftime("%m.%d")
    log_dir = f"./logs/{current_time}"
    model_dir = f"./model/{current_time}"
    
    check_dir(log_dir)
    check_dir(model_dir)
    
    return log_dir, model_dir

def select_gpu():
    """
    Selects the GPU with the least memory usage.

    Returns:
        tuple: (device, memory_usage_mb) where device is a torch.device and memory_usage_mb is a float.
    """
    if not torch.cuda.is_available():
        logging.info("No GPU available, using CPU.")
        return torch.device("cpu"), 0.0

    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        if num_gpus == 0:
            logging.info("No GPU available, using CPU.")
            return torch.device("cpu"), 0.0

        # Query memory usage for each GPU
        gpu_memory = []
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory.append(info.used)

        # Select GPU with minimal memory usage
        selected_gpu = torch.argmin(torch.tensor(gpu_memory)).item()
        memory_mb = gpu_memory[selected_gpu] / 1024 ** 2
        logging.info(f"Selected GPU {selected_gpu} with memory usage: {memory_mb:.2f} MB")
        return torch.device(f"cuda:{selected_gpu}"), round(memory_mb, 2)

    except Exception as e:
        logging.warning(f"Failed to query GPU memory: {e}. Using default GPU.")
        return torch.device("cuda:0"), 0.0
    finally:
        pynvml.nvmlShutdown()