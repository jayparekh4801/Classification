import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

class LogisticRegressionCreator :
    def __init__(self, file) :
        if(type(file) == type(str(file))) :
            self.dataset = pd.read_csv("data.csv")
            self.featureMatrix = self.dataset.iloc[ : , : -1].values
            self.resultMatrix = self.dataset.iloc[ : , -1].values
        
        elif(type(file) == type(np.array)) :
            self.dataset = file
            self.featureMatrix = file[ : , : -1]
            self.resultMatrix = file[ : , -1]
        
        self.preProcessing()
    
    def preProcessing(self) :
        scaler = StandardScaler()
        self.featureMatrix_train, self.featureMatrix_test, self.resultMatrix_train, self.resultMatrix_test = train_test_split(self.featureMatrix, self.resultMatrix, test_size=0.2, random_state=0)
        self.featureMatrix_train_sca = scaler.fit_transform(self.featureMatrix_train)
        self.featureMatrix_test_sca = scaler.transform(self.featureMatrix_test)
        self.makeModel()

    def makeModel(self) :
        self.classifier = LogisticRegression(random_state = 0)
        self.classifier.fit(self.featureMatrix_train_sca, self.resultMatrix_train)
        self.predictor()
    
    def predictor(self) :
        self.resultMatrix_pred = self.classifier.predict(self.featureMatrix_test_sca)
        # accuracyCounter()
    
    def accuracyCounter(self) :
        print(confusion_matrix(self.resultMatrix_test, self.resultMatrix_pred))
        return accuracy_score(self.resultMatrix_test, self.resultMatrix_pred)

