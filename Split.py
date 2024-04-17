import os
import shutil
import random

def split_and_save_data(data_folder, output_folder='split_data', test_size=0.2, random_state=42):
    # Create output folder for the split data
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of class folders
    class_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

    # Iterate through each class folder
    for class_folder in class_folders:
        # Get the list of image files in the class folder
        class_files = [os.path.join(data_folder, class_folder, file) for file in os.listdir(os.path.join(data_folder, class_folder)) if file.endswith('.jpg') or file.endswith('.png')]

        # Split the files into training and testing sets
        class_train_files, class_test_files = train_test_split(class_files, test_size=test_size, random_state=random_state)

        # Create output folders for train and test sets within the class folder
        train_folder_path = os.path.join(output_folder, 'train', class_folder)
        test_folder_path = os.path.join(output_folder, 'test', class_folder)
        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(test_folder_path, exist_ok=True)

        # Copy training files to the train folder
        for file_path in class_train_files:
            shutil.copy(file_path, os.path.join(train_folder_path, os.path.basename(file_path)))

        # Copy testing files to the test folder
        for file_path in class_test_files:
            shutil.copy(file_path, os.path.join(test_folder_path, os.path.basename(file_path)))

    return os.path.join(output_folder, 'train'), os.path.join(output_folder, 'test')

def create_folder(data_folder):
    class_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]
    print(class_folders)
    for i in class_folders:
        new_folder=os.path.join(data_folder,"test",i)
        print(new_folder)
        os.makedirs(new_folder, exist_ok=True)

def move_to_foler(data_dir):
    class_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]
    class_folders.remove("Test")
    class_folders.remove("Train")
    print(class_folders)
    for class_name in class_folders:
        train_dir = os.path.join(data_dir, class_name)
        os.rmdir(train_dir)
        

# Specify the path to your data folder
data_folder = 'C:/code/pytorch_course/plant_disease/Data'

# Split the data into training and testing sets and save the sets to new directories
#train_folder, test_folder = split_and_save_data(data_folder)
move_to_foler(data_folder)
# The train_folder and test_folder variables contain the paths to the new directories