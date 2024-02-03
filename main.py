import torch
import numpy as np
import idx2numpy
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms
from torchvision.datasets import mnist

transform = transforms.ToTensor()

train_filepaths = ["./ImageClassification/archive/train-images.idx3-ubyte", "./ImageClassification/archive/train-labels.idx1-ubyte"] # Index 0 for images, 1 for labels
test_filepaths = ["./ImageClassification/archive/t10k-images.idx3-ubyte", "./ImageClassification/archive/t10k-labels.idx1-ubyte"] # Index 0 for images, 1 for labels

# train_img_filepath = "./ImageClassification/archive/train-images.idx3-ubyte"
# train_label_filepath = "./ImageClassification/archive/train-labels.idx1-ubyte"
# test_img_filepath = "./ImageClassification/archive/t10k-images.idx3-ubyte"
# test_label_filepath = "./ImageClassification/archive/t10k-labels.idx1-ubyte"

# Step 1, import all the images into NumPy arrays
# Step 2, Normalise the datapoints (RGB [0-255]) (Image is already in grayscale so only 1 channel)
# Step 3, Convert to tensors
# Step 3, Flatten the matrix, -> turn into 1D array.

# def import_images(filepath):
#     images = idx2numpy.convert_from_file(filepath)
#     images_tensor = transforms.ToTensor()(images.copy()).permute(1, 0, 2) # This turns the numpy array into a torch tensor and also normalizes values between 0-1.
#     return images_tensor

# def import_labels(filepath):
#     labels = idx2numpy.convert_from_file(filepath)
#     labels_tensor = torch.tensor(labels.copy(), dtype=torch.int64)
#     return labels_tensor

# print(test_images[:, 3, :])
# print(train_images[: , 3, :])
# print(train_images.shape, train_labels.shape)

# Load the data
# train_dataset = TensorDataset(train_images, train_labels)
# test_dataset = TensorDataset(test_images, test_labels)


# def import_image_and_labels(img_filepath, label_filepath):
#     images = idx2numpy.convert_from_file(img_filepath)
#     labels = idx2numpy.convert_from_file(label_filepath)
#     images_tensor = transforms.ToTensor()(images.copy()).permute(1, 0, 2)

#     return images_tensor, labels

# def preprocessor(image_filepath,label_filepath, batch_size):
#     images, labels = import_image_and_labels(image_filepath, label_filepath)

#     dataset = TensorDataset(images, labels)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     return loader


# test_loader = preprocessor(train_img_filepath, test_label_filepath, batch_size=32)


# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# conv_layer1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1)
# conv_layer2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1)
# pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
# fully_connected_layer1 = torch.nn.Linear(in_features=40*7*7, out_features=120)
# fully_connected_layer2 = torch.nn.Linear(in_features=120, out_features=10)



# Create class version of the neural network

class MNISTModel(torch.nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1, stride=1) # Padding is 1 so output size is the same as input
        self.conv_layer2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1, stride=1) 
        self.pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2) # Halves the dimensions, 28x28 -> 14x14
        self.fully_connected1 = torch.nn.Linear(in_features=40*7*7, out_features=120) # We end up with 40 filters applied to a 7x7 image. Flattening out gives 40*7*7
        self.fully_connected2 = torch.nn.Linear(in_features=120, out_features=10) # Will use a softmax output so we want 10 outputs (10 possible digits)

    def forward(x):
        x = pooling_layer(torch.relu(conv_layer1(x)))# 28 -> 14
        x = pooling_layer(torch.relu(conv_layer2(x))) # 14 -> 7

        x = torch.flatten(x, 1)

        x = torch.relu(fully_connected_layer1(x))
        x = fully_connected_layer2(x)
        x = torch.softmax(x, dim=1)

        return x
    
class Trainer():
    def __init__(self, model, optimizer, epochs, batch_size, learning_rate, train_filepaths, test_filepaths, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # self.train_loader = train_loader
        # self.test_loader = test_loader
        self.train_img_filepath, self.train_label_filepath = train_filepaths
        self.test_img_filepath, self.test_label_filepath = test_filepaths  
        self.loss_func = loss_function

    def import_image_and_labels(img_filepath, label_filepath):
        images = idx2numpy.convert_from_file(img_filepath)
        labels = idx2numpy.convert_from_file(label_filepath)
        images_tensor = transforms.ToTensor()(images.copy()).permute(1, 0, 2)

        return images_tensor, labels

    def preprocessor(self, image_filepath,label_filepath):
        images, labels = import_image_and_labels(image_filepath, label_filepath)

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return loader


    def train(self):

        # loss_func = torch.nn.CrossEntropyLoss()
        self.train_loader = preprocessor(self.train_img_filepath, self.train_label_filepath)
        self.test_loader = preprocessor(self.test_img_filepath, self.test_label_filepath)
        optimizer = torch.optim.Adam(self.model.parameters(), learning_rate=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_training_loss = 0

            for batch_inputs, batch_labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.loss_func(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_training_loss += loss.item()

        return self.model
    
    def eval_model(self):
        self.model.eval()
        total_loss = 0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for batch_inputs, batch_labels in self.test_loader:
                outputs = self.model(batch_inputs)
                loss = self.loss_func(outputs, batch_labels)
                total_loss += loss.item() * batch_inputs.size(0) # Multiply by batch size since we're calculating loss over each batch?
                _, pred_labels = torch.max(outputs, 1) # Just extracting the labels from output.
                correct_preds += (pred_labels == batch_labels).sum().item()
                total_preds += batch_labels.size(0)

        avg_loss = total_loss / len(self.test_loader.dataset) #?
        accuracy = correct_preds / total_preds

        return accuracy, avg_loss


