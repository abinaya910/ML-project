import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot as plt
%matplotlib inline
iris=pd.read_csv('C:\\Users\\Desktop\\datasets_19_420_Iris.csv')
iris.head()
iris.describe()
print(iris.shape)
print(iris.groupby('Species').size())# label distribution
#univariant plots- box & whisker plots
iris.plot(kind='box',subplots=True,layout=(10,10),sharex=False,sharey=False)
plt.show()
iris.hist()
plt.show()
#multi-variant
scatter_matrix(iris)
plt.show()
# validation set
array=iris.values
X=array[:,1:5]
Y=array[:,5]
X_train,X_Validation,Y_train,Y_Validation=train_test_split(X,Y,test_size=0.2,random_state=1)
#LOGISTIC REGRESSION
#LINEAR DISCRIMINANT ANALYSIS,KNN,SVM
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
results=[]
names=[]
for name, model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1)#10fold validation
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s:%f (%f)' %(name,cv_results.mean(),cv_results.std()))
 #compare the models
plt.boxplot(results, labels=names)
plt.title('Comparison')
plt.show()
model=SVC(gamma='auto')
model.fit(X_train,Y_train)
pred=model.predict(X_Validation)
print(accuracy_score,pred)
print(confusion_matrix(Y_Validation,pred))
print(classification_report(Y_Validation,pred))
