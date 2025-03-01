import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

plt.style.use("ggplot")


############### Dataset ###############
def plot_column_hist(df, column_name, title=None):
    plt.figure(figsize=(10, 5))
    plt.hist(df[column_name], bins=30, color="skyblue")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    if title:
        plt.title(title)


def plot_column_boxplot(df, column_name, title=None):
    plt.figure(figsize=(10, 5))
    plt.boxplot(df[column_name])
    plt.ylabel(column_name)
    if title:
        plt.title(title)
    plt.show()


def plot_column_scatter(df, x_column, y_column, title=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)


def plot_column_hist_with_normal_dist(df, column_name, title=None):
    plt.figure(figsize=(10, 5))
    data = df[column_name]
    mu, std = stats.norm.fit(data)
    plt.hist(data, bins=30, density=True, alpha=0.6, color="skyblue")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)
    plt.xlabel(column_name)
    plt.ylabel("Density")
    if title:
        plt.title(title)
    plt.show()


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


############### Model ###############


def plot_loss_function(training_loss, validation_loss=None, title="Loss Function"):
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label="Training Loss")
    if validation_loss is not None:
        plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
