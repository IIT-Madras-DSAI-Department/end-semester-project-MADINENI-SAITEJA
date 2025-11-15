import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# --------------------------------------------------------
#                READ MNIST DATA  (YOUR FORMAT)
# --------------------------------------------------------
def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    if 'even' in featurecols:
        featurecols.remove('even')
    Xtrain = dftrain[featurecols].values
    ytrain = dftrain['label'].values
    Xval = dfval[featurecols].values
    yval = dfval['label'].values
    return Xtrain, ytrain, Xval, yval



#XGB



import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from collections import Counter
import random
import time
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


start=time.time()

# ------------------------
#  UPDATED READ_DATA
# ------------------------
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



# --------------------------------------------------------------------
# YOUR ORIGINAL BINARY XGBOOST (UNCHANGED)
# --------------------------------------------------------------------
class XGBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.5, max_depth=7, 
                 reg_lambda=1.0, gamma=0.0, 
                 n_bins=256, 
                 subsample=0.8, 
                 colsample_bytree=0.5):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []
        self.bin_thresholds_ = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _gradients(self, y_true, y_pred):
        p = self._sigmoid(y_pred)
        g = p - y_true
        h = np.maximum(p * (1 - p), 1e-16)
        return g, h

    def _create_bins(self, X):
        self.bin_thresholds_ = {}
        for j in range(X.shape[1]):
            try:
                thresholds = np.quantile(
                    X[:, j], 
                    q=np.linspace(0, 1, self.n_bins + 1)[1:-1],
                )
                self.bin_thresholds_[j] = np.unique(thresholds)
            except ValueError:
                self.bin_thresholds_[j] = np.unique(X[:, j])
        
    def _apply_bins(self, X):
        X_binned = np.empty_like(X, dtype=np.int32)
        for j in range(X.shape[1]):
            X_binned[:, j] = np.digitize(X[:, j], self.bin_thresholds_[j])
        return X_binned

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.trees = []

        self._create_bins(X)
        X_binned = self._apply_bins(X)
        y_pred = np.zeros(n_samples)

        for t in range(self.n_estimators):
            if self.subsample < 1.0:
                sample_idx = np.random.choice(
                    n_samples, 
                    size=int(n_samples * self.subsample), 
                    replace=False
                )
            else:
                sample_idx = np.arange(n_samples)
            
            g, h = self._gradients(y[sample_idx], y_pred[sample_idx])
            
            tree = self._fit_tree(
                X_binned[sample_idx], 
                X[sample_idx],
                g, 
                h, 
                depth=0
            )
            self.trees.append(tree)
            y_pred += self.learning_rate * self._predict_tree(tree, X)

    def _gain(self, G, H):
        if H + self.reg_lambda < 1e-6:
            return 0.0
        return 0.5 * (G ** 2) / (H + self.reg_lambda)

    def _fit_tree(self, X_binned, X_orig, g, h, depth=0):
        n_samples, n_features = X_binned.shape

        if depth >= self.max_depth or n_samples <= 2:
            w = -np.sum(g) / (np.sum(h) + self.reg_lambda)
            return {'leaf': True, 'value': w}

        best_gain, best_split, best_feature = 0, None, None
        G_total, H_total = np.sum(g), np.sum(h)
        base_gain = self._gain(G_total, H_total)

        if self.colsample_bytree < 1.0:
            features_idx = np.random.choice(
                n_features,
                size=int(n_features * self.colsample_bytree),
                replace=False
            )
        else:
            features_idx = np.arange(n_features)

        for j in features_idx:
            hist_g = np.bincount(X_binned[:, j], weights=g, minlength=self.n_bins)
            hist_h = np.bincount(X_binned[:, j], weights=h, minlength=self.n_bins)
            G_left, H_left = 0.0, 0.0
            for i in range(self.n_bins - 1):
                G_left += hist_g[i]
                H_left += hist_h[i]
                if H_left < 1e-6:
                    continue
                G_right = G_total - G_left
                H_right = H_total - H_left
                if H_right < 1e-6:
                    continue
                gain = self._gain(G_left, H_left) + self._gain(G_right, H_right) - base_gain - self.gamma
                if gain > best_gain:
                    best_gain, best_split_idx, best_feature = gain, i, j

        if best_feature is None:
            w = -np.sum(g) / (np.sum(h) + self.reg_lambda)
            return {'leaf': True, 'value': w}

        best_thr = self.bin_thresholds_[best_feature][best_split_idx]
        left_idx = X_binned[:, best_feature] <= best_split_idx
        right_idx = ~left_idx
        
        left = self._fit_tree(
            X_binned[left_idx], X_orig[left_idx], 
            g[left_idx], h[left_idx], 
            depth + 1
        )
        right = self._fit_tree(
            X_binned[right_idx], X_orig[right_idx],
            g[right_idx], h[right_idx],
            depth + 1
        )

        return {
            'leaf': False, 
            'feature': best_feature, 
            'threshold': best_thr, 
            'left': left, 
            'right': right
        }

    def _predict_tree(self, tree, X):
        if tree['leaf']:
            return np.full(X.shape[0], tree['value'])
        feature, threshold = tree['feature'], tree['threshold']
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        out = np.empty(X.shape[0])
        out[left_idx] = self._predict_tree(tree['left'], X[left_idx])
        out[right_idx] = self._predict_tree(tree['right'], X[right_idx])
        return out

    def predict_proba(self, X):
        X = np.asarray(X)
        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.learning_rate * self._predict_tree(tree, X)
        return self._sigmoid(pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)



# --------------------------------------------------------------------
#  MULTI-CLASS WRAPPER (One-vs-Rest)
# --------------------------------------------------------------------
class OneVsRestXGB:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = []

    def fit(self, X, y):
        self.models = []
        for cls in range(10):
            print(f"Training Class {cls} ...")
            y_binary = (y == cls).astype(int)
            model = XGBoostClassifier(**self.kwargs)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        # get proba for each of 10 models
        prob_matrix = np.vstack([m.predict_proba(X) for m in self.models]).T
        # pick highest probability → predicted digit
        return np.argmax(prob_matrix, axis=1)
# --------------------------------------------------------------------
# TRAIN  (0–9 CLASSIFICATION)
# --------------------------------------------------------------------
Xtrain, ytrain, Xval, yval = read_data()

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


