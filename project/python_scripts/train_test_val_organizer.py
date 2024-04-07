"""
Script used to moved dog photo files into separate Train, Test, and Validation folders within a structure
"""
import os
import shutil
from math import ceil

INPUT_DIR = "/Users/redge/Library/CloudStorage/OneDrive-UniversityofCalgary/School/MEng/Winter2024/enel645/my-645/645-project/tests/dataset-143-classes/"

def split_and_move_files(source_folder, destination_folder, split_ratio):
    # Get all the files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Calculate split indices
    split_index_1 = ceil(len(files) * split_ratio[0])
    split_index_2 = split_index_1 + ceil(len(files) * split_ratio[1])
    
    # Split files according to the defined ratio
    train_files = files[:split_index_1]
    test_files = files[split_index_1:split_index_2]
    validation_files = files[split_index_2:]
    
    # Function to move files
    def move_files(files, source, destination):
        os.makedirs(destination, exist_ok=True)
        for file in files:
            shutil.move(os.path.join(source, file), destination)

    # Move files to the respective directories
    move_files(train_files, source_folder, os.path.join(destination_folder, 'Train', source_folder.replace(INPUT_DIR, "")))
    move_files(test_files, source_folder, os.path.join(destination_folder, 'Test', source_folder.replace(INPUT_DIR, "")))
    move_files(validation_files, source_folder, os.path.join(destination_folder, 'Validation', source_folder.replace(INPUT_DIR, "")))

def process_directory(input_directory):
    for root, dirs, files in os.walk(input_directory, topdown=True):
        
        dirs[:] = [d for d in dirs if d not in ['Train', 'Test', 'Validation']]
        for dir_name in dirs:
            print(f"root = {root}")
            source_folder = os.path.join(root, dir_name)
            split_and_move_files(source_folder, root, (0.7, 0.2, 0.1))
            print(f"source_folder.replace(INPUT_DIR, "")}...   ...Source folder moved")

# Usage
input_directory = INPUT_DIR
process_directory(input_directory)