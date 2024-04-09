import torch
from torch import nn
from torchvision import models


def train(model, optimizer, criterion, metric, data, epochs):
    train_loader = data["train"]
    valid_loader = data["valid"]

    if torch.cuda.is_available():
        model.to("cuda")
        metric.to("cuda")

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    for epoch in range(epochs):

        # Pongo el modelo en modo entrenamiento
        model.train()

        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0

        for train_data, train_target in train_loader:

            if torch.cuda.is_available():
                train_data = train_data.to("cuda")
                train_target = train_target.to("cuda")

            optimizer.zero_grad()
            output = model(train_data.float())
            loss = criterion(output, train_target.float())
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            accuracy = metric(output, train_target)
            epoch_train_accuracy += accuracy.item()

        epoch_train_loss = epoch_train_loss / len(train_loader)
        epoch_train_accuracy = epoch_train_accuracy / len(train_loader)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_accuracy)

        # Pongo el modelo en modo testeo
        model.eval()

        epoch_valid_loss = 0.0
        epoch_valid_accuracy = 0.0

        for valid_data, valid_target in valid_loader:
            if torch.cuda.is_available():
                valid_data = valid_data.to("cuda")
                valid_target = valid_target.to("cuda")

            output = model(valid_data.float())
            epoch_valid_loss += criterion(output, valid_target.float()).item()
            epoch_valid_accuracy += metric(output, valid_target.float()).item()

        epoch_valid_loss = epoch_valid_loss / len(valid_loader)
        epoch_valid_accuracy = epoch_valid_accuracy / len(valid_loader)
        valid_loss.append(epoch_valid_loss)
        valid_acc.append(epoch_valid_accuracy)

        print("Epoch: {}/{} - Train loss {:.6f} - Train Accuracy {:.6f} - Valid Loss {:.6f} - Valid Accuracy {:.6f}".
              format(epoch + 1, epochs, epoch_train_loss, epoch_train_accuracy, epoch_valid_loss, epoch_valid_accuracy))

    history = {}
    history["train_loss"] = train_loss
    history["train_acc"] = train_acc
    history["valid_loss"] = valid_loss
    history["valid_acc"] = valid_acc

    return history


# Models definition
class FacesSimpleCNN(nn.Module):
    def __init__(self):
        super(FacesSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        # self.fc2 = nn.Linear(512, 2)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.final = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.final(x)
        return x.squeeze()


class ResNet18Binary(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        # Pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # Replace the final fully-connected layer with a binary classifier
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)
        self.final = torch.nn.Sigmoid()

    def forward(self, x):
        # # Pass the input through the ResNet18 model
        # features = self.resnet(x)
        #
        # # Determine the flattened size dynamically
        # flattened_size = features.numel()  # Number of elements in the tensor
        #
        # # Flatten the features using the correct size
        # features = features.view(-1, flattened_size)
        #
        # # Apply the binary classifier
        # output = self.fc(features)
        # output = self.final(output)

        x = self.resnet(x)
        output = self.final(x)

        return output.squeeze()
