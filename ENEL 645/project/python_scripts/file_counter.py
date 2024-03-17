"""
file counter v0.1
"""
import os
import pandas as pd

INPUT_DIR = "/Users/redge/Library/CloudStorage/OneDrive-UniversityofCalgary/School/MEng/Winter2024/enel645/my-645/645-project/tests/dataset-143-classes/"

def count_files_in_directory(input_directory):
    folder_counts = {}

    for root, dirs, files in os.walk(input_directory):
        if root != input_directory:
            folder_counts[root] = len(files)
    
    return folder_counts

def create_dataframe(folder_counts):
    print(list(folder_counts))
    df = pd.DataFrame(list(folder_counts.items()), columns=['Folder', 'File Count'])
    # Remove preceding text (e.g., 'text_to_remove' followed by digits and a slash)
    df['Folder'] = df['Folder'].str.replace(F'{INPUT_DIR}/', '')
    

    return df

# Input directory path
input_directory = INPUT_DIR

# Count the files in the directory
folder_counts = count_files_in_directory(input_directory)

# Create a DataFrame from the folder counts
df = create_dataframe(folder_counts)
df.sort_values(by='Folder', inplace=True)

# Display the DataFrame
display(df)
print(df.describe())

cell_value = df[df['File Count'] > 50]
print(cell_value)