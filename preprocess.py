from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

def splitData(x, y, perc):
    if perc > 1 and perc < 100:
        return train_test_split(x, y, test_size=perc/100)
    elif perc > 0 and perc < 1:
        return train_test_split(x, y, test_size=perc)
    else:
        return None
def polyFeatures(data, degree=1):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    data = poly.fit_transform(data)
    return data