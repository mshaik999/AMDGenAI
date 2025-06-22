import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
iris=load_iris()
X=iris.data
Y=iris.target

X_train, X_test, Y_train, Y_Test=train_test_split(X,Y,test_size=0.2,random_state=42)

lg=LogisticRegression(max_iter=300)
lg.fit(X_train,Y_train)
joblib.dump(lg,'iris_model')