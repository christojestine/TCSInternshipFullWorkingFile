import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')
data=pd.read_csv(r"D:\ICT DSA\TCS Internship\Project Deployment\Preprocessed Dataset.csv")
# The best Moded we have recived as per the jupyter notebook data processing and model building is KNN model
X=data.drop('salary',axis=1)
Y=data['salary']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.20)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score;
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=17)
classifier=classifier.fit(X_train,Y_train)
y_pred_KNN=classifier.predict(X_test)
acc=accuracy_score(Y_test,y_pred_KNN)
print("Confusion Matrix")
print(confusion_matrix(Y_test,y_pred_KNN))
print("Classification Report")
print(classification_report(Y_test,y_pred_KNN))
print('Accuracy Score=',accuracy_score(Y_test,y_pred_KNN))
print('F1 Score <=50K=',f1_score(Y_test,y_pred_KNN,pos_label='<=50K'))
print('F1 Score >50K=',f1_score(Y_test,y_pred_KNN,pos_label='>50K'))
pickle.dump(classifier,open('model.pkl','wb'))
print("Checking Whether the Model.pkl file is created or not")
model=pickle.load(open('model.pkl','rb'))
prediction=model.predict(X_test)
print(accuracy_score(Y_test,prediction))