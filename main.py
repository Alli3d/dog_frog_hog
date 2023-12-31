import numpy as np
import cv2
import os
import pywt
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle

# testing cv2.imread() is working correctly
try:
    img1 = cv2.imread('test_images/hog1.jpeg')
except NameError or ImportError:
    print("OpenCV has failed")
else:
    print("OpenCV test: successful")

def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H


def categorise_with_pic_names(dir):
    allpics = (os.walk(dir))
    for pic in allpics:
        return pic[2]


category = {
    "dog": categorise_with_pic_names('picdata/dog/'),
    "frog": categorise_with_pic_names('picdata/frog/'),
    "hog": categorise_with_pic_names('picdata/hog/')
}


def scaling_transforming(img, marker):
    scaledraw = cv2.resize(img, (32, 32))
    img_trans = w2d(img, 'db1', 5)
    scaledtrans = cv2.resize(img_trans, (32, 32))

    combined = np.vstack((scaledraw.reshape(32 * 32 * 3, 1), scaledtrans.reshape(32 * 32, 1)))
    x.append(combined)
    y.append(marker)


x = []
y = []
for i in category["dog"]:
    img = cv2.imread('picdata/dog/' + i)
    scaling_transforming(img, 0)
print("Dog images cleaned")

for i in category["frog"]:
    img = cv2.imread('picdata/frog/' + i)
    scaling_transforming(img, 1)
print("Frog images cleaned")

for i in category["hog"]:
    img = cv2.imread('picdata/hog/' + i)
    scaling_transforming(img, 2)
print("Hog images cleaned")

x = np.array(x).reshape(len(x), 4096).astype(float)
x_train, x_test, y_train, y_test = train_test_split(x, y)

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    }
}

scores = []

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

print("Algorithm + parameter sweep results:")
for results in scores:
    print(results)

# training the best algorithm (manually selected from above results)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear', C=1))])
pipe.fit(x_train, y_train)

print("\n\nClassification report and confusion matrix for best combination:")
print(classification_report(y_test, pipe.predict(x_test)))
cm = confusion_matrix(y_test, pipe.predict(x_test))
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

with open('dogfroghog.pickle', 'wb') as f:
    pickle.dump(pipe, f)
