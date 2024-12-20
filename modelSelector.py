from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

class Model(ABC):
    model = None

    @abstractmethod
    def __init__(self):
        pass
    #@abstractmethod
    #def __init__(self, model):
    #    pass
        #if model == "linear" :
        #    self.model = None
    @abstractmethod
    def fit(self, data, out):
        pass
    @abstractmethod
    def predict(self, data):
        pass

class LinearModel(Model):
    def __init__(self):
        self.model = LinearRegression()
    def fit(self, data, out):
        self.model.fit(data,out)
    def predict(self, data):
        return self.model.predict(data)

#class PolynomialModel(Model):
#    def __init__(self):
#        self.model = LinearRegression()
#    def fit(self, exp, data, out):
#        poly = PolynomialFeatures(order=exp, include_bias=False)
#        data = poly.fit_transform(data)
#        self.model.fit(data, out)
#    def predict(self, data):
#        return self.model.predict(data)

class DecisionTree(Model):
    def __init__(self):
        self.model = DecisionTreeClassifier()
    def fit(self, data, out):
        self.model.fit(data, out)
    def predict(self, data):
        return self.model.predict(data)
