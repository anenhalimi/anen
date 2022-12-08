# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
data = []
with open('chronic_kidney_disease.arff', "r") as f:
    for line in f:
        line = line.replace('\n', '')
        data.append(line.split(','))


names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
         'bgr', 'bu',  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
         'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane',
         'class', 'no_name']
    
df = pd.DataFrame(data[29:429], columns=names)

print(df.head())

df = df.replace('?', np.nan)
df = df.replace('	?', np.nan)
df[["age", "bp", "sg", "al", "su",'bgr', 'bu',  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
         'rbcc']] = df[["age", "bp", "sg", "al", "su",'bgr', 'bu',  'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
         'rbcc']].apply(pd.to_numeric)
df.info()            
df['rbc'] = df['rbc'].map({'normal': 1, 'abnormal': 0})
df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
df['pcc'] = df['pcc'].map({'present': 1, 'notpresent': 0})
df['ba'] = df['ba'].map({'present': 1, 'notpresent': 0})
df['htn'] = df['htn'].map({'yes': 1, 'no': 0})
df['dm'] = df['dm'].map({'yes': 1, 'no': 0,' yes' : 1,'\tyes' : 1,'\tno' : 0})
df['cad'] = df['cad'].map({'yes': 1, 'no': 0,'\tno' : 0})
df['appet'] = df['appet'].map({'good': 1, 'poor': 0,'no' : 0})
df['ane'] = df['ane'].map({'yes': 1, 'no': 0})
df['pe'] = df['pe'].map({'yes': 1, 'no': 0,'good' : 1})
df['class'] = df['class'].map({'ckd': 1, 'notckd': 0,'ckd\t' : 1,'no' : 0})
df = df.drop(columns="no_name",axis=1)
names.remove("no_name")
df.isnull().values.any()
df["rbc"] = df["rbc"].fillna(df["rbc"].mode()[0])
df['pc'] = df['pc'].fillna(df['pc'].mode()[0])
df['pcc'] = df['pcc'].fillna(df['pcc'].mode()[0])
df['ba'] = df['ba'].fillna(df['ba'].mode()[0])
df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
df['dm'] = df['dm'].fillna(df['dm'].mode()[0])
df['cad'] = df['cad'].fillna(df['cad'].mode()[0])
df['appet'] = df['appet'].fillna(df['appet'].mode()[0])
df['ane'] = df['ane'].fillna(df['ane'].mode()[0])
df['class'] = df['class'].fillna(df['class'].mode()[0])
df['cad'] = df['cad'].fillna(df['cad'].mode()[0])
df['pe'] = df['pe'].fillna(df['pe'].mode()[0])
df['class'] = df['class'].fillna(df['class'].mode()[0])
df.isna().sum().sum()
from sklearn.impute import KNNImputer
imput = KNNImputer(n_neighbors = 5)
df = imput.fit_transform(df)
df
df = pd.DataFrame(df ,columns = names)
df
df.isna().sum().sum()
import matplotlib.pyplot as plt
def fonction_DA(y,df):
    Q1 = np.percentile(df[y], 25, interpolation = 'midpoint')
    Q2 = np.percentile(df[y], 50, interpolation = 'midpoint')  
    Q3 = np.percentile(df[y], 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    for x in df[y].index:
        if (df[y][x] >= up_lim):
            df[y][x] = up_lim
        elif (df[y][x] <= low_lim):
            df[y][x] = low_lim 
def fonction_KNN(y,df):
    
    Q1 = np.percentile(df[y], 25, interpolation = 'midpoint')
    Q2 = np.percentile(df[y], 50, interpolation = 'midpoint')  
    Q3 = np.percentile(df[y], 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    for x in df[y].index:
        if (df[y][x] >= up_lim) or (df[y][x] <= low_lim):
            df[y][x]=np.nan
            fonction_KNN("age",df)
fonction_DA("bp",df)
fonction_DA("sg",df)
fonction_DA("al",df)
fonction_DA("bgr",df)
fonction_DA("bu",df)
fonction_DA("sc",df)
fonction_DA("sod",df)
fonction_DA("pot",df)
fonction_DA("hemo",df)
fonction_KNN("pcv",df)
fonction_DA("wbcc",df)
fonction_KNN("rbcc",df)
from sklearn.impute import KNNImputer
imput = KNNImputer(n_neighbors = 5)
df['age'] = imput.fit_transform(df['age'].values.reshape(-1, 1))
df['rbcc'] = imput.fit_transform(df['rbcc'].values.reshape(-1, 1))
df['pcv'] = imput.fit_transform(df['pcv'].values.reshape(-1, 1))
for item in df.index:
    df.loc[item,"gfr"]=186*((df.loc[item,"sc"])**(-1.154))*df.loc[item,"age"]**(-0.203)*0.94
df['gfr']
names.append('gfr')
for item in df.index:
    if df.loc[item,"gfr"] > 90:
        df.loc[item,"stade"] = 1
    elif df.loc[item,"gfr"] < 89 and df.loc[item,"gfr"] > 60 :
        df.loc[item,"stade"] = 2
    elif df.loc[item,"gfr"] < 59 and df.loc[item,"gfr"] > 30 :
        df.loc[item,"stade"] = 3
    elif df.loc[item,"gfr"] < 29 and df.loc[item,"gfr"] > 15 :
        df.loc[item,"stade"] = 4
    else :
        df.loc[item,"stade"] = 5
df["stade"]
names.append('stade')
df = df.drop(columns="gfr",axis=1)
names.remove("gfr")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns = names)
df
df = df.drop(columns="sod",axis=1)
names.remove("sod")
df = df.drop(columns="pe",axis=1)
names.remove("pe")
df = df.drop(columns="bp",axis=1)
names.remove("bp")
df = df.drop(columns="bu",axis=1)
names.remove("bu")
df = df.drop(columns="age",axis=1)
names.remove("age")
df = df.drop(columns="appet",axis=1)
names.remove("appet")
df = df.drop(columns="wbcc",axis=1)
names.remove("wbcc")
df = df.drop(columns="su",axis=1)
names.remove("su")
df = df.drop(columns="pot",axis=1)
names.remove("pot")
df = df.drop(columns="pc",axis=1)
names.remove("pc")
df = df.drop(columns="rbc",axis=1)
names.remove("rbc")
df = df.drop(columns="ane",axis=1)
names.remove("ane")
df = df.drop(columns="ba",axis=1)
names.remove("ba")
df = df.drop(columns="pcc",axis=1)
names.remove("pcc")
df = df.drop(columns="cad",axis=1)
names.remove("cad")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

target_col="class"
X = df.drop('class',axis=1)
y = df['class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, weights='uniform')
knn_model.fit(X_train,y_train)
y_pred_rfc = knn_model.predict(X_test)
score_rfc=accuracy_score(y_pred_rfc,y_test)*100
score_rfc=accuracy_score(y_pred_rfc,y_test)*100
score_rfc1=f1_score(y_pred_rfc,y_test)*100
score_rfc2=precision_score(y_pred_rfc,y_test)*100
score_rfc3=recall_score(y_pred_rfc,y_test)*100
print("LinearDiscriminant Classifier SCORE:{:.3f}".format(score_rfc))
print("LinearDiscriminant Classifier f1_score:{:.3f}".format(score_rfc1))
print("LinearDiscriminant Classifier precision:{:.3f}".format(score_rfc2))
print("LinearDiscriminant Classifier recall:{:.3f}".format(score_rfc3))
print("Logistic Regression Classifier SCORE:{:.3f}".format(score_rfc))
















import pickle
pickle.dump(knn_model,open('model.pkl','wb'))

knn_model = pickle.load(open('model.pkl','rb'))
print(knn_model.predict([[0.75,0.2,0.3,0.15,0.8,0.7,0.6,1,1,0.25]]))
df.columns
X






























































