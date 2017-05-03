import pickle
import numpy as np

class Cifar10Loader():
    """
    Dataset: CIFAR 10
    Results with CNN: 18% test error without data augmentation and 11% with.
    Jasper Snoek has a new paper in which he used Bayesian hyperparameter optimization to find
    nice settings of the weight decay and other hyperparameters, which allowed him to obtain a
    test error rate of 15% (without data augmentation) using the architecture of the net that got 18%.
    """
    def __init__(self):
        self.train_data_files = ['data_batch_1', 'data_batch_2' ,'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.test_data_file  = 'test_batch'
        self.img_size = 32
        self.num_channels = 3
        self.img_size_flat = self.img_size * self.img_size * self.num_channels
        self.num_classes = 10
        self.images_per_file = 10000

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def _get_data_label_from_file(self, file):
        data_dict = self._unpickle('/'.join([self.dataset_path, file]))
        return data_dict[b'data'], np.array(data_dict[b'labels'])

    def load_cifar10(self, data_path='../Datasets/cifar-10'):
        """
        Parameters
        ----------
        data_path (str)
            Path to cifar10 dataset.
        """
        self.dataset_path = data_path
        for i, train_data_file in enumerate(self.train_data_files):
            if i == 0:
                X_train, y_train = self._get_data_label_from_file(train_data_file)
            else:
                data, label = self._get_data_label_from_file(train_data_file)
                X_train = np.vstack((X_train, data))
                y_train = np.hstack((y_train, label))
        X_train = X_train.reshape(X_train.shape[0], self.num_channels, self.img_size, self.img_size).transpose(0, 2, 3, 1).astype("float")
        X_train = np.array(X_train, dtype=float) / 255.0

        X_test, y_test = self._get_data_label_from_file(self.test_data_file)
        X_test = X_test.reshape(X_test.shape[0], self.num_channels, self.img_size, self.img_size).transpose(0, 2, 3, 1).astype("float")
        X_test = np.array(X_test, dtype=float) / 255.0

        labels_dict = self._unpickle('/'.join([self.dataset_path, "batches.meta"]))
        labels = [bytes.decode(a) for a in labels_dict[b'label_names']]
        return X_train, y_train, X_test, y_test, labels
