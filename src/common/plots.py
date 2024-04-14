import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import seaborn as sns


def plot_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(history["train_loss"])
    axs[0].plot(history["valid_loss"])
    axs[0].title.set_text('Error de Entrenamiento vs Validación')
    axs[0].legend(['Train', 'Valid'])

    axs[1].plot(history["train_acc"])
    axs[1].plot(history["valid_acc"])
    axs[1].title.set_text('Accuracy de Entrenamiento vs Validación')
    axs[1].legend(['Train', 'Valid'])


def show_transformed_images(train_dataset, train_dataset_aug):
    images_ids = np.random.randint(low=0, high=len(train_dataset), size=4)
    fig, rows = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))

    # Plot the images without augmentation
    for id, row in enumerate(rows[0]):
        row.imshow(train_dataset[images_ids[id]][0].permute(1, 2, 0))
        row.axis('off')

    # Plot the same images but with augmentation
    for id, row in enumerate(rows[1]):
        row.imshow(train_dataset_aug[images_ids[id]][0].permute(1, 2, 0))
        row.axis('off')

    plt.show()


def plot_predictions(model,
                     valid_loader,
                     label_map={0: 'fake', 1: 'real'}):
    plt.figure(figsize=(10, 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels = next(iter(valid_loader))

    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    # _, predicted = torch.max(outputs, 1)
    predicted = outputs
    predicted_binary = [1 if value > 0.5 else 0 for value in predicted]

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        color = 'green' if predicted_binary[i] == labels[i] else 'red'
        plt.title(f'Actual: {label_map[int(labels[i])]}, Predicted: {label_map[predicted_binary[i]]}', color=color)
        plt.axis('off')

    plt.show()


def evaluate_model(model, valid_loader, metric):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs, 1)
            predicted = outputs
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # print(f'Accuracy: {accuracy_score(all_labels, all_preds)}')
    # metric_result = metric(all_preds, all_labels)
    metric_result = metric(torch.tensor(all_preds), torch.tensor(all_labels))
    print(f'Accuracy: {metric_result}')

    # report
    all_preds_binary = [1 if value > 0.5 else 0 for value in all_preds]

    print("Classification Report:")
    print(classification_report(all_labels, all_preds_binary))

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds_binary)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def roc_plot_comparison(model_array, valid_loader):
    plt.figure(figsize=(8, 8))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    all_preds = []
    all_labels = []

    for model, label in model_array:
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # _, predicted = torch.max(outputs, 1)
                predicted = outputs
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        fpr, tpr, _ = roc_curve(torch.tensor(all_labels), torch.tensor(all_preds))
        plt.plot(fpr, tpr, label=label)

    plt.xlabel("False positives ratio")
    plt.ylabel("True positives ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()
