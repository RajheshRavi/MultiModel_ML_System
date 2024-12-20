from flask import Flask, request 
import pandas as pd

import modelSelector as mod                     # Module containing the ML models
import preprocess as pp                         # Module to Pre-Process the data
import errorCalc as er                          # Module to Calculate Error

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return {"message":"Hello World!!"}

@app.route("/fit/", methods=["GET","POST"])
def predict():
    modl = str(request.form.get("model"))
    polyFlag = False
    if "linear" in modl.lower():
        model = mod.LinearModel()
    elif "poly" in modl.lower():
        model = mod.LinearModel()
        polyFlag = True
    elif "decision" in modl.lower() and "tree" in modl.lower():
        model = mod.DecisionTree()
    else:
        return "Choose one of the available models"
    dataset = request.files.get("dataSet")
    if dataset.content_type == "text/csv":
        dataset = pd.read_csv(dataset)
    elif dataset.content_type == "application/vnd.ms-excel" or dataset.conennt_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        dataset = pd.read_excel(dataset)
    else:
        return "Choose the Correct Data-Set file!!  \nThe Current file type is: " + dataset.content_type 
    x = dataset.iloc[:,0:-1]
    y = dataset.iloc[:,-1]
    if polyFlag and request.form.get("degree"):
        pp.polyFeatures(x, round(float(request.form.get("degree"))))
    x_train, x_test, y_train, y_test = pp.splitData(x,y, float(request.form.get("testSplit")))
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    return str(er.errorCalc(y_test=y_test, y_pred=y_pred))



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")