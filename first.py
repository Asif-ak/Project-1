import numpy as np
import pandas as pd
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical

seed=7
np.random.seed(seed)
#print(a)

dataframe=pd.read_csv("sonar.csv",header=None)
dataset=dataframe.values
X=dataset[:,0:60].astype(float) # its is getting rows and columns that y there is a ','
Y=dataset[:,60]                 # its is getting rows and columns that y there is a ','
#print(X)
#print("++++++++++++++++++++++++")
#print(Y)
print(dataframe.groupby(60).size())
def BaseLine_Model():
    model=models.Sequential()
    model.add(layers.Dense(60, activation='relu',input_shape=(X.shape[1],)))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

estimator=KerasClassifier(build_fn=BaseLine_Model, epochs=100,batch_size=10,verbose=0)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
result=cross_val_score(estimator,X,Y,cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))
