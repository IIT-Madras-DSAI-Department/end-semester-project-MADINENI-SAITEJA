import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from collections import Counter
import random
import time
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from algorithms import XGBoostClassifier,OneVsRestXGB

def read_data(trainfile='MNIST_train.csv', validationfile='mnist_test.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')   # keep all pixels
    # remove "even" only if present
    if 'even' in featurecols:
        featurecols.remove('even')

    Xtrain = dftrain[featurecols]
    ytrain = dftrain['label']     # <--- MAIN CHANGE (0–9 target)
    
    Xval = dfval[featurecols]
    yval = dfval['label']

    return (Xtrain.values, ytrain.values, Xval.values, yval.values)

Xtrain, ytrain, Xval, yval = read_data()
start=time.time()
model = OneVsRestXGB(
    n_estimators=30,
    learning_rate=0.5,
    max_depth=5,
    reg_lambda=1.0,
    gamma=0.0
)

model.fit(Xtrain, ytrain)

end=time.time()

ypred = model.predict(Xval)
print("Time Taken", end-start)
print("Accuracy:", accuracy_score(yval, ypred))
cm = confusion_matrix(yval, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
print("Precision:", precision_score(yval, ypred, average='macro'))
print("f1 score:", f1_score(yval, ypred, average='macro'))
ytrain_pred = model.predict(Xtrain)

train_error = 1 - accuracy_score(ytrain, ytrain_pred)
val_error   = 1 - accuracy_score(yval, ypred)

# Simple interpretable bias–variance terms
bias_estimate = train_error
variance_estimate = val_error - train_error   # how much error increases on validation
noise_estimate = max(0, val_error - bias_estimate - variance_estimate)

print("\n=============== BIAS–VARIANCE BREAKDOWN ===============")
print(f"Training Error (Bias)         : {bias_estimate:.4f}")
print(f"Generalization Gap (Variance) : {variance_estimate:.4f}")
print(f"Noise (Unavoidable Error)     : {noise_estimate:.4f}")
print("=======================================================\n")
plt.show()
