import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale, OneHotEncoder, OrdinalEncoder
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import DenseFeatures, Dense, Dropout
import os

'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col = 'PassengerId')

#Name, ticket number, cabin number, and port of embarkation can be assumed irrelevant to predicting survival outcome
data = data.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)



#Creating column that represents whether a passenger was alone(1) or not(0)
alone = []
for (i, j) in zip(data['SibSp'], data['Parch']):
    if i + j == 0:
        alone.append(1)
    else:
        alone.append(0)
        
data['Alone'] = alone

cols = list(data.columns.values)
cols[-1] = 'Fare'
cols[-2] = 'Alone'
data = data[cols]

data['Age'] = data['Age'].astype('float64') #----------------------------
for i in data:
    if data[i].dtype == 'int64':
        #print('h')
        data[i] = data[i].astype('float64')
        


X = data.drop(['Survived'], axis=1)
y = data['Survived']


cols.pop(0)

#print(data.isna().sum())
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
imputer = SimpleImputer(strategy='most_frequent')

transformer = ColumnTransformer([('Imputer', imputer, cols),
                                 #('One Hot Encoder', OneHotEncoder(handle_unknown='ignore', sparse=False), ['Sex']) #Encoder raises an error
                                ])
oh = OrdinalEncoder()

#Preprocessing on X training data
transformed_X_train = pd.DataFrame(transformer.fit_transform(X_train))
OH_X_train = pd.DataFrame(oh.fit_transform(transformed_X_train))
OH_X_train.index = transformed_X_train.index
#print(transformed_X_train.head())
#num_X_train = transformed_X_train.drop([1], axis=1)
#OH_X_train = pd.concat([num_X_train, OH_X_train], axis=1)
final_X_train = pd.DataFrame(scale(OH_X_train))


#Preprocessing on X validation data
transformed_X_val = pd.DataFrame(transformer.transform(X_val))  #Impute
print(transformed_X_val.head())
OH_X_val = pd.DataFrame(oh.fit_transform(transformed_X_val))   #Encode
print(OH_X_val.head())
OH_X_val.index = transformed_X_val.index
print(OH_X_val.head())
#num_X_val = transformed_X_val.drop([1], axis=1)
#OH_X_val = pd.concat([num_X_val, OH_X_val], axis=1)
#print(OH_X_val.head())
final_X_val = pd.DataFrame(scale(OH_X_val))


class TitanicModel(Model):
    
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.drop1 = Dropout(0.25)
        self.d3 = Dense(128, activation='relu')
        self.d4 = Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.drop1(x)
        x = self.d3(x)
        return self.d4(x)
    
    
model = TitanicModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(np.asarray(final_X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64), epochs=10, verbose =2, validation_data=(np.asarray(final_X_val), np.asarray(y_val)))

#preds = model.predict(final_X_val)
#predictions = []
#print(preds[0][0])
'''for i in preds:
    predictions.append(round(i[0], 0))

print(f"First five actual labels: {y_val.head()}")
print(f"First five predictions: {predictions[0:5]}")
#print(X_val.head())
print(y_val.head())
#model.evaluate(np.asarray(final_X_val), np.asarray(y_val))'''