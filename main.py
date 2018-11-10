import numpy as np
import sklearn as sk
import pandas as pd
import utils

# set the random seed
np.random.seed(566)

# import data
all_data = pd.read_csv("all-data.csv")


# split the data into test, train, and validation sets
test_size = 0.2 # percent of the dataset that should be test

# try oversampling the minority class

# try undersampling the majority class
