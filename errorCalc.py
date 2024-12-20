from sklearn.metrics import root_mean_squared_error

def errorCalc(y_test, y_pred):
    return root_mean_squared_error(y_pred=y_pred, y_true=y_test)