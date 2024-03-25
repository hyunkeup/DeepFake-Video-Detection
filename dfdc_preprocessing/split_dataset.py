import os
import shutil
import numpy as np

# Set the path to your datasets
real_folder = "../datasets/raw_dataset/raw_video_audio_pairs/real"
fake_folder = "../datasets/raw_dataset/raw_video_audio_pairs/fake"
destination_folder = "../datasets/raw_dataset/split_dataset"

# Set the split ratios for train, validation, and test sets
split_ratios = (0.7, 0.2, 0.1)  # Ensure these add up to 1

# Create destination subfolders if they don't exist
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(destination_folder, split), exist_ok=True)

# List and count real data pairs
real_data_pairs = [f[:-4] for f in os.listdir(real_folder) if f.endswith('.mp4')]
num_real_data_pairs = len(real_data_pairs)

# List fake data pairs and randomly select the same number as real data pairs
fake_data_pairs = [f[:-4] for f in os.listdir(fake_folder) if f.endswith('.mp4')]
selected_fake_data_pairs = np.random.choice(fake_data_pairs, num_real_data_pairs, replace=False)


# Function to split and copy files
def split_and_copy_files(data_pairs, source_folder):
    # Shuffle data pairs to randomize the distribution
    np.random.shuffle(data_pairs)

    # Calculate split sizes
    num_train = int(len(data_pairs) * split_ratios[0])
    num_val = int(len(data_pairs) * split_ratios[1])

    # Split data pairs
    train_data = data_pairs[:num_train]
    val_data = data_pairs[num_train:num_train + num_val]
    test_data = data_pairs[num_train + num_val:]

    # Copy function
    def copy_files(data_subset, split_name):
        for data_name in data_subset:
            for ext in ['.mp4', '.wav']:
                src_path = os.path.join(source_folder, data_name + ext)
                dest_path = os.path.join(destination_folder, split_name, data_name + ext)
                shutil.copy(src_path, dest_path)

    # Copy files to respective directories
    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    copy_files(test_data, 'test')

# Split and copy real data
split_and_copy_files(real_data_pairs, real_folder)

# Split and copy selected fake data
split_and_copy_files(selected_fake_data_pairs, fake_folder)

print('Dataset split and copy completed.')