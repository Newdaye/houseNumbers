#required
import sklearn
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from svhn import SVHN
#SKlearn Modules
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Loads the .mat file where the data is located and then applies
def loadData(file):
    data = loadmat(file)
    xData = data['X']
    yData = data['y']
    xData, yData = xData.transpose((3, 0, 1, 2)), yData[:, 0]
    print("Loaded: ", file)
    print("Feature: ", xData.shape)
    print("Target: ", yData.shape)
    return xData, yData
#svhn = SVHN("", 10, use_extra=False, gray=True)

if __name__ == "__main__":
    xTrain, yTrain = loadData('train_32x32.mat')
    xTest, yTest = loadData('test_32x32.mat')
    xTrain = xTrain.reshape(73257,3*32*32)
    xTest = xTest.reshape(26032,3*32*32)
    #xTrain = np.reshape(xTrain, (73257,32,32,3))
    #xTest = np.reshape(xTest, (26032,32,32,3))

    xTrain = np.divide(xTrain, 255)
    xTest = np.divide(xTest, 255)


    print(xTrain.shape)
    print(xTest.shape)

    xTrain, xTest, yTrain, yTest = train_test_split(xTrain, yTrain, test_size=.2, random_state=4)


    #hidden layer = (25,11,7,5,3)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(25,11,7,5,3),
                        max_iter = 50, activation = 'logistic',
                        solver = 'sgd')

    sc = StandardScaler()
    scaler = sc.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    mlp_clf.fit(xTrain, yTrain)

    mlp_clf.score(xTest, yTest)

    yPred = mlp_clf.predict(xTest)

    print("Accuracy: {:.2f}".format(accuracy_score(yTest, yPred)))
    print(classification_report(yTest, yPred))

    plt.plot(mlp_clf.loss_curve_)
    plt.title("loss curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()