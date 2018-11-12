import numpy as np
import sklearn as sk
import pandas as pd
import utils
from sk.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

from sklearn.naive_bayes import BernouilliNB


# set the random seed
np.random.seed(566)

## Prepare the data
# import data
all_data = pd.read_csv("all-data.csv")
p = all_data.shape[1] - 1 # number of features
majority_data = all_data.loc[, "inactive"]
n_minority = all_data.loc[, "active"].shape[0] # number of minority samples

# split the data into test, train, and validation sets
test_size = 0.3 # percent of the dataset that should be set aside for test
n_splits = 10 # number of splits for cross-validation
# just doing a regular train/test
train_X, train_y, test_X, test_y = train_test_split(all_data[, 1:p], undersampled[, -1], test_size = test_size, shuffle = True)
kf = KFold(n_splits = n_splits)
kf.get_splits(train_X)

# try oversampling the minority class
# NOTE: Should only oversample on the training data to ensure independence between training and test sets
# Use SMOTE to ensure that we are not overfitting?
# Test set might have radically different distribution than train set
#sm = SMOTE(ratio = 1.0)
#oversampled_data =
#oversampled_train_X, oversampled

# try undersampling the majority class
# get as many majority samples as n_minority
# both train and test sets should be relatively balanced
undersampled_data = majority_data.sample(n = n_minority)
undersampled_train_X, undersampled_train_y, undersampled_test_X, undersampled_test_y = train_test_split(undersampled_data[, 1:p], undersampled[, -1], test_size = test_size, shuffle = True)
undersampled_kf = KFold(n_splits = n_splits)
undersampled_kf.get_splits(undersampled_train_X)



## Perform the experiments
