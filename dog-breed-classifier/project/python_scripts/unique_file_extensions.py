"""
Display all unique file extensions in a folder.
"""
import os

INPUT_DIR = "/Users/redge/Library/CloudStorage/OneDrive-UniversityofCalgary/School/MEng/Winter2024/enel645/my-645/645-project/tests/dataset-143-classes"

def find_unique_file_extensions(input_directory):
    unique_extensions = set()

    # Recursive function to traverse directories and find file extensions
    def search_extensions(directory):
        for entry in os.scandir(directory):
            if entry.is_file():
                _, ext = os.path.splitext(entry.name)
                if ext:  # Check if there is an extension
                    unique_extensions.add(ext)
            elif entry.is_dir():
                search_extensions(entry.path)

    search_extensions(input_directory)
    return list(unique_extensions)

# Input directory path
input_directory = INPUT_DIR

# Find unique file extensions in the directory
extensions = find_unique_file_extensions(input_directory)

# Output the list of unique file extensions
print(extensions)