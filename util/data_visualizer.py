import numpy as np
import matplotlib.pyplot as plt

class DataVisualizer():
    def __init__(self):
        pass

    def visualize_data(self, X, title='examples'):
        fig, axes1 = plt.subplots(5,5,figsize=(3,3))
        plt.suptitle(title)
        for j in range(5):
            for k in range(5):
                i = np.random.choice(range(len(X)))
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X[i:i+1][0])

    def visualize_data_of_class(self, X, y, class_id, title='examples'):
        fig, axes1 = plt.subplots(5,5,figsize=(3,3))
        plt.suptitle(title)
        indexes = np.array(np.where(y == class_id)).flatten()
        for j in range(5):
            for k in range(5):
                i = np.random.choice(indexes)
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X[i:i+1][0])

    def visualize_image(self, image, title='example'):
        plt.figure(figsize=(2,2))
        plt.title(title)
        plt.imshow(image, interpolation="spline16")