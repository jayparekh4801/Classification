from Logistic_Regression import LogisticRegressionCreator
from KNN_Classifier import KNNClassifierCreator

log = LogisticRegressionCreator("data.csv")
print("Accuracy Of Lgistic Regression is " + str(log.accuracyCounter()))
print("----------------------------------------------------------------------")
knn = KNNClassifierCreator("data.csv")
print("Accuracy Of KNN Classifier is " + str(knn.accuracyCounter()))