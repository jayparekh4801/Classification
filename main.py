from Logistic_Regression import LogisticRegressionCreator
from KNN_Classifier import KNNClassifierCreator
from SVM_Classifier import SVMClassifierCreator
from Kernel_SVM import SVMKernelClassifierCreator

log = LogisticRegressionCreator("data.csv")
print("Accuracy Of Lgistic Regression is " + str(log.accuracyCounter()))
print("----------------------------------------------------------------------")
knn = KNNClassifierCreator("data.csv")
print("Accuracy Of KNN Classifier is " + str(knn.accuracyCounter()))
print("----------------------------------------------------------------------")
svm = SVMClassifierCreator("data.csv")
print("Accuracy Of SVM Classifier is " + str(svm.accuracyCounter()))
print("----------------------------------------------------------------------")
svmK = SVMKernelClassifierCreator("data.csv")
print("Accuracy Of SVM Classifier is " + str(svmK.accuracyCounter()))