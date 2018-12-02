import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

#import algorithms

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale

#from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import KFold

#from sklearn.naive_bayes import BernouilliNB


# set the random seed
np.random.seed(566)

## Prepare the data
## Prepare the data
# import data
all_data = pd.read_csv("all-data.csv")
all_data['Output'] = all_data.apply(lambda r: 1 if r['Outcome'] == 'Active' else 0, axis = 1)
all_data.drop('Outcome', inplace = True, axis = 1)



# split the data into test and train sets
test_size = 0.25 # percent of the dataset that should be set aside for test
n_splits = 5 # number of splits for cross-validation
X_data, y_data = all_data.loc[:, all_data.columns != 'Output'], all_data['Output']
X_data = scale(X_data) # scale the data since the features have different magnitudes

train_X, test_X, train_y, test_y = train_test_split(X_data, y_data, test_size = test_size, shuffle = True)
#print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

p = all_data.shape[1] - 1 # number of features

majority_data = all_data.loc[all_data["Output"] == 0]
n_minority = all_data.loc[all_data["Output"] == 1].shape[0] # number of minority samples
print("Total samples = " + str(all_data.shape[0]))
print("There are " + str(n_minority) + " minority samples.")


# cross-validation for the regular train/test split
#kf = StratifiedKFold(n_splits = n_splits)
#kf.get_n_splits(train_X)

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
"""undersampled_data = majority_data.sample(n = n_minority)
undersampled_train_X, undersampled_train_y, undersampled_test_X, undersampled_test_y = train_test_split(undersampled_data[:, 1:p], undersampled[:, -1], test_size = test_size, shuffle = True)
undersampled_kf = KFold(n_splits = n_splits)
undersampled_kf.get_splits(undersampled_train_X)"""



# Run the experiments
# cross-validation experiments

# cross-validation parameters
regwgts = np.arange(1, 6, 1)
kernels = ["linear", "rbf"]
gammas = np.arange(0.2, 1.1, 0.2)
lr_params = {"C":regwgts}
svm_params = {"gamma": gammas}

# Already ran CV, so commented it out
"""print("Starting logistic regression CV")
lr = LogisticRegression(penalty = "l1", solver = "saga", random_state = None, class_weight = "balanced", max_iter = 90)
lr_cv = GridSearchCV(lr, lr_params, cv = n_splits, scoring = "f1")
lr_cv.fit(train_X, train_y)
print(lr_cv.best_params__)
print(lr_cv.cv_results__)"""
C_best = 2#lr_cv.best_params__["C"]

"""print("Starting SVM CV")
svm = SVC(kernel = "rbf", random_state = None, class_weight = "balanced", max_iter = 1000)
svm_cv = GridSearchCV(svm, svm_params, cv = n_splits, scoring = "f1")
svm_cv.fit(train_X, train_y)
print(svm_cv.best_params__)
print(svm_cv.cv_results__)"""
gamma_best = 1.0#svm_cv.best_params__["gamma"]


# final experiments (e.g., to get standard error)
numruns = 8

# try a neural network since svm can take too long to converge
nn = MLPClassifier(hidden_layer_sizes = (16, 8), alpha = 0.0, max_iter = 10, random_state = None)

final_algs = {
    "Logistic Regression": LogisticRegression(penalty = "l1", solver = "saga", random_state = None, class_weight = "balanced", max_iter = 90, C = C_best),
    "SVM": SVC(kernel = "rbf", random_state = None, class_weight = "balanced", gamma = gamma_best, max_iter = 1000),
    "Naive Bayes": BernoulliNB(alpha = 1.0, fit_prior = True)
    #"Neural Network": nn
    }

print("Starting final experiments")



conf_mats = {} # holds the confusion matrices for each algorithm
f1 = {} # holds the list of macro f1 scores for each algorithm
for name in final_algs.keys():
    conf_mats[name] = pd.DataFrame([[0, 0], [0, 0]])
    f1[name] = []

# compute macro average of f1 score (i.e., f1 score for every run) so that we may calculate a confidence interval
for i in range(numruns):
    # randomize the data set
    print("run #" + str(i))
    train_X, test_X, train_y, test_y = train_test_split(X_data, y_data, test_size = test_size, shuffle = True)
    for name, alg in final_algs.items():
        print("running " + name)
        alg.fit(train_X, train_y)
        pred_y = alg.predict(test_X)
        mat = pd.crosstab(test_y, pred_y)
        #print(mat)
        tp = mat.iloc[1, 1]
        fp = mat.iloc[0, 1]
        fn = mat.iloc[1, 0]
        precision = tp / (tp + fp)
        print(precision)
        if precision < 0.00001:
        	precision = 0.00001
        recall = tp / (tp + fn)
        if recall < 0.00001:
        	recall = 0.00001
        f1[name].append(2 * precision * recall / (precision + recall))
        conf_mats[name] = conf_mats[name].add(mat)
        print(conf_mats[name])

# plot errors/do significance test to understand if errors are actually normally distributed
# also find confidence intervals
for name in final_algs.keys():
    # plot the histogram
    plt.hist(sorted(f1[name]))
    plt.title("F1 error for " + name)
    plt.show()

    # show a QQ plot
    sp.stats.probplot(scale(f1[name]), plot = plt)
    plt.title("QQ plot for F1 error of " + name)
    plt.show()

    # do the scipy normality test
    statistic, p_value = sp.stats.normaltest(f1[name])
    print("Normal statistic for " + name + " = " + str(statistic) + ", p = " + str(p_value))

    # assuming scores are normally distributed, calculate 95% confidence interval
    mu = np.mean(f1[name])
    sigma = np.std(f1[name])
    left = mu - 1.96 * sigma / np.sqrt(numruns)
    right = mu + 1.96 * sigma / np.sqrt(numruns)
    print("95% confidence interval for " + name + ": " + "(" + str(left) + ", " + str(right) + ")")

# print the confusion matrices for each algorithm and calculate the avg f1 score
for name, mat in conf_mats.items():
    print("Confusion matrix for " + name + ": ")
    print(mat)
    f1_std_err = np.std(f1[name]) / np.sqrt(numruns)
    print("Avg F1 score for " + name + ": " + str(np.mean(f1[name])) + "+-" + str(1.96 * f1_std_err))
