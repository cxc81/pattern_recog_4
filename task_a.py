import numpy as np
import gzip

# Set the path to the downloaded files
data_path = "./"


# Function to extract the images from the MNIST files
def extract_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return data


# Function to extract the labels from the MNIST files
def extract_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# Extract the images and labels from the downloaded files
train_images = extract_images(data_path + "train-images-idx3-ubyte.gz")
train_labels = extract_labels(data_path + "train-labels-idx1-ubyte.gz")
test_images = extract_images(data_path + "t10k-images-idx3-ubyte.gz")
test_labels = extract_labels(data_path + "t10k-labels-idx1-ubyte.gz")

# Save the images and labels as .npy files
np.save(data_path + "train_images.npy", train_images)
np.save(data_path + "train_labels.npy", train_labels)
np.save(data_path + "test_images.npy", test_images)
np.save(data_path + "test_labels.npy", test_labels)

print("Conversion completed and files saved.")
