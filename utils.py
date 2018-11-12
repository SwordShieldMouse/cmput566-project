def cross_validate_error(alg, X, y, kf):
    # returns cross-validation error
    # assumes we have been given an alg from scikit learn, so that we can call fit() and predict()
    total_error = 0
    for train_ix, test_ix in kf.split(X):
        alg.fit(X[train_ix], y[train_ix])
        preds = alg.predict(X[test_ix])
        total_error += sum([a == b for a, b in zip(y[test_ix], preds)])
    return total_error
