import numpy as np
import os
from sklearn.model_selection import train_test_split


class Utils:

    def __init__(self, desc=None):
        self.description = desc
        self.X = None
        self.Y = None

    def read_data(self, file_path):
        try:

            features = []
            labels = []

            dir_name = os.path.join(os.path.dirname(file_path), 'numpy')

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            if os.stat(os.path.join(dir_name, "features.npy")).st_size == 0:

                f = open(file_path, 'r')
                for idx, line in enumerate(f):

                    # jump the meta data in the first line
                    if idx != 0:
                        row = line.split(',')  # this will return array of the form [label , pixels , training/testing]
                        labels.append(row[0])
                        features.append([int(pixel) for pixel in row[1].split()])

                # finally save
                np.save(os.path.join('features.npy', np.array(features)))  # .npy extension is added if not given
                np.save(os.path.join('labels.npy', np.array(labels)))  # .npy extension is added if not given

                self.X = features
                self.Y = labels

            else:
                self.X = np.load(os.path.join('features.npy'))  #
                self.Y = np.load(os.path.join('labels.npy'))    #

        except Exception as e:
            print(e)

    # load and prepocess data
    def process_data(self):
        """
        This model loads the MNIST dataset from the keras api
        @ return : x_train : training features
                   y_train : training labels
                   x_test  : test features
                   y_test : test labels
        """
        num_classes = 10

        # load dataset

        print('y_train.shape=', self.X.shape)
        print('y_test.shape=', self.Y.shape)

        # Convert class vectors to binary class matrices ("one hot encoding")
        ## Doc : https://keras.io/utils/#to_categorical
        # y_train = tensorflow.keras.utils.to_categorical(y_train)
        # y_test = tensorflow.keras.utils.to_categorical(y_test)

        # Convert to float
        # x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')
        # print('x_train.shape=', x_train.shape)
        # print('x_test.shape=', x_test.shape)
        #
        # # Normalize inputs from [0; 255] to [0; 1]
        # x_train = x_train / 255
        # x_test = x_test / 255
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8)
        #
        # return x_train, y_train, x_test, y_test, x_val, y_val

    def print_class_distribution(self, class_label, number_of_samples):

        """
        Prints a given class label and number of instances

        """
        print("Class Label {} contains {} samples".format(class_label, number_of_samples))
        return self

    def split_data(self, X_all, y_all):
        """

        :param X: Input data that contains the feature data
        :return:
        """
        # keras with tensorflow backend
        N, D = X_all.shape
        X_all = X_all.reshape(N, 48, 48, 1)

        # Split in  training set : validation set :  testing set in 80:10:10
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=0)
        y_train = (np.arange(len(np.unique(y_all))) == y_train[:, None]).astype(np.float32)
        y_test = (np.arange(len(np.unique(y_all))) == y_test[:, None]).astype(np.float32)

        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # create utils class
    util = Utils()

    filename = '../dataset/fer2013.csv'
    X, Y = util.read_data(filename)

    print("Total labels = {}".format(len(np.unique(Y))))
    classes, number_of_sample = np.unique(Y, return_counts=True)

    # to see how many sample each class contain use the helper function
    [util.print_class_distribution(class_label, samples) for class_label, samples in
     zip(list(classes), list(number_of_sample))]
