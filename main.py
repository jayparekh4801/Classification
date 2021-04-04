from Logistic_Regression import LogisticRegressionCreator
from KNN_Classifier import KNNClassifierCreator
from SVM_Classifier import SVMClassifierCreator
from Kernel_SVM import SVMKernelClassifierCreator
from Naive_Baise_Classifier import NaiveBayesClassifierCreator
from Decision_Tree_Classifier import DecisionTreeClassifierCreator
from Random_forest_Tree_Classifier import RandomForestClassifierCreator

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
print("Accuracy Of SVM_Kernel Classifier is " + str(svmK.accuracyCounter()))
print("----------------------------------------------------------------------")
naiveB = NaiveBayesClassifierCreator("data.csv")
print("Accuracy Of Naive Bayes Classifier is " + str(naiveB.accuracyCounter()))
print("----------------------------------------------------------------------")
decT = DecisionTreeClassifierCreator("data.csv")
print("Accuracy Of Decision Tree Classifier is " + str(decT.accuracyCounter()))
print("----------------------------------------------------------------------")
ranT = RandomForestClassifierCreator("data.csv")
print("Accuracy Of Decision Tree Classifier is " + str(ranT.accuracyCounter()))