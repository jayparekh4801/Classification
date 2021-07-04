###########################################NOTE###################################################################
##### Make Sure That Your Data Has Not Categorical Data In It And Also Make Sure That Class Column Should Be The #
##### Last Column Of Data                                                                                        #
##################################################################################################################


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
knn = KNNClassifierCreator("heart.csv")
print("Accuracy Of KNN Classifier is " + str(knn.accuracyCounter()))
print("----------------------------------------------------------------------")
svm = SVMClassifierCreator("heart.csv")
print("Accuracy Of SVM Classifier is " + str(svm.accuracyCounter()))
print("----------------------------------------------------------------------")
svmK = SVMKernelClassifierCreator("heart.csv")
print("Accuracy Of SVM_Kernel Classifier is " + str(svmK.accuracyCounter()))
print("----------------------------------------------------------------------")
naiveB = NaiveBayesClassifierCreator("heart.csv")
print("Accuracy Of Naive Bayes Classifier is " + str(naiveB.accuracyCounter()))
print("----------------------------------------------------------------------")
decT = DecisionTreeClassifierCreator("heart.csv")
print("Accuracy Of Decision Tree Classifier is " + str(decT.accuracyCounter()))
print("----------------------------------------------------------------------")
ranT = RandomForestClassifierCreator("heart.csv")
print("Accuracy Of Decision Tree Classifier is " + str(ranT.accuracyCounter()))