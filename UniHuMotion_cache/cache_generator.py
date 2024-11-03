import argparse
import numpy as np
import random
import torch
from dataset import create_dataset



def generate_data_cache(chunk_size=4, dataset_name=""):

    # print('Generate training data cache...')
    # dataset_train = create_dataset(dataset_name, split="train")

    print('Generate validation data cache...')
    dataset_val = create_dataset(dataset_name, split="val")

    # print('Generate test data cache...')
    # dataset_test = create_dataset(dataset_name, split="test")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--UniHuMotion_dataset", type=str, default="NBA", help="Name of the dataset with UniHuMotion framework")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = args.UniHuMotion_dataset

    
    generate_data_cache(dataset_name=dataset)








