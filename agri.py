import pandas as pd
ad = pd.read_csv("agridata.csv")
ad

df = pd.read_csv("agridata2.csv")
df

inputs = df.drop('results',axis='columns')
target = df['results']
df["outlook"].fillna("No_data", inplace = True) 
df["temp"].fillna("No_data", inplace = True) 
df["windy"].fillna("No_data", inplace = True) 
df

from sklearn.preprocessing import LabelEncoder
le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_windy = LabelEncoder()
le_soil = LabelEncoder()


inputs['outlook_n'] = le_outlook.fit_transform(inputs['outlook'])
inputs['temp_n'] = le_temp.fit_transform(inputs['temp'])
inputs['windy_n'] = le_windy.fit_transform(inputs['windy'])
inputs['soil_n'] = le_soil.fit_transform(inputs['soil'])
inputs

inputs_n = inputs.drop(['outlook','temp','windy','soil'],axis='columns')
inputs_n

target

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

model.score(inputs_n,target)

model.predict([[2,1,1,1]])

model.predict([[2,0,0,0]])
