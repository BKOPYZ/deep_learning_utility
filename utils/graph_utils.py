import matplotlib.pyplot as plt
import numpy as np


def plot_show_image_with_labels(row, col, images, labels, title=None):
    fig, axs = plt.subplots(row, col, figsize=(15, 15))
    for i in range(row):
        for j in range(col):
            axs[i, j].imshow(images[i * col + j])
            axs[i, j].set_title(labels[i * col + j])
            axs[i, j].axis("off")
    if title:
        fig.suptitle(title)
    plt.show()


def plot_compare_image(row, col, images, labels, title=None):
    figure = plt.figure(figsize=(15, 15))
    for i in range(row * col):
        figure.add_subplot(row, col, i + 1)
        plt.imshow(images[i])
        plt.imshow(labels[i], alpha=0.5)
        plt.title(f"image: {i}")
    plt.show()
