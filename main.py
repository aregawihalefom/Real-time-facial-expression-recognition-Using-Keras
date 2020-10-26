import numpy as np
from models.Model1 import Network1
from utils.utils import *

if __name__ == '__main__':
    # 1. HyperParameteres
    #-----------------------------
    batch_size = 128
    epochs = 124
    num_classes = 7
    filename = 'dataset/fer2013.csv'
    # ------------------------------

    # 2. Load and preporcess data
    # ----------------------------------
    util = Utils()

    # read data
    util.read_data(filename)

    # process data
    util.process_data()

    # split data
    X_train, y_train, X_test, y_test = util.split_data(X, y)

    # 3. Create model and train
    # --------------------------------

    # create model
    model1 = Network1(num_classes=7, desc="Model with  3 Cov and 3 Dense Layers")
    model = model1.create_model()

    # train model
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1111)




