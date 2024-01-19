import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Create a Transformer
class Frequency_Transformer_Single():
    
    def __init__(self, name_f, name_Label):
        self.name_f = name_f
        self.Label = name_Label
        self.nf_name = name_f+"_ave"
        
    def fit(self, X, y=None):
        self.group = X.groupby(by = self.name_f)[self.Label].mean()
        self.med = np.nanmean(self.group.values)
        return X
 
        
    def transform(self, X, y=None):
        X[self.nf_name] = X[self.name_f].map(self.group)
        if X[self.nf_name].isna().sum()==0:
            return X
        else:
            if X[self.name_f].dtype=='float64':
                return self.numeric_fill(X, y)
            else:
                X[self.nf_name].fillna(self.med, inplace=True)
                return X   
               
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def numeric_fill(self, X, y=None):
        nan_indices = X[self.nf_name].isna()
        values_in_feature2 = X.loc[nan_indices, self.name_f]
        list_a=[]
        num = 10
        for x in values_in_feature2.values:
            list_a.append(self.find_near(x,num))
        
        X[self.nf_name][nan_indices]=list_a
        return X
        
    def find_near(self, x, num):
        A_group = np.array(self.group.index)
        diff = np.abs(A_group-x)
        indices = np.argpartition(diff,num)[:num]
        nearest_values=A_group[indices]
        x = self.group[nearest_values].sum()/num
        return x

class Feq_Transformer_Multi():
    
    def __init__(self, names, label):
        self.transformers = []
        for name in names:
            fq = Frequency_Transformer_Single(name, label)
            self.transformers.append(fq)        
        
    def fit(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.fit(X)
        return X
            
    def transform(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def fit_transform(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.fit(X)
        return self.transform(X)
            
    def numeric_fill(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.numeric_fill(X)
        return X
    
    def find_near(self, X, num):
        for transformer in self.transformers:
            X = transformer.find_near(X)
        return X

        
#load the original data
df = pd.read_csv("fraud_payment_data.csv")

#To avoid AttributeError: 'NoneType' object has no attribute 'groupby' (Still having this problem)
df['Sender_Id'] = df.Sender_Id.fillna(df['Bene_Id'])
df['Bene_Id'] = df.Bene_Id.fillna(df['Sender_Id'])

df['Sender_Account'] = df.Sender_Account.fillna(df['Bene_Account'])
df['Bene_Account'] = df.Bene_Account.fillna(df['Sender_Account'])

df['Sender_Country'] = df.Sender_Country.fillna(df.Bene_Country)
df['Bene_Country'] = df.Bene_Country.fillna(df.Sender_Country)

df['Sender_Sector'] = df.Sender_Sector.fillna('Unknown')
df['Sender_Sector'] = df.Sender_Sector.astype('str')

#Define key features for model training after EDA
features = ['SDAYPair', 'USD_amount', 'Sender_Sector', 'Bene_Account']
target = 'Label'

#Feature engineering
df['Time_step']=pd.to_datetime(df.Time_step)
df['dayofyear'] = df.Time_step.dt.dayofyear
df['SDAYPair']=(df['Sender_Id'] + '-' + df['dayofyear'].astype('str'))

#Define key features for model training after EDA
names = ['SDAYPair', 'USD_amount', 'Sender_Sector', 'Bene_Account']
features = ['SDAYPair_ave', 'USD_amount_ave', 'Sender_Sector_ave', 'Bene_Account_ave']
target = 'Label'

#Train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42) 
X_train_raw, y_train = train[['SDAYPair', 'USD_amount', 'Sender_Sector', 'Bene_Account', 'Label']], train[target]
X_test_raw,  y_test  = test[['SDAYPair', 'USD_amount', 'Sender_Sector', 'Bene_Account', 'Label']], test[target]

# List of features and their corresponding transformers 
encoder = Feq_Transformer_Multi(names, target)
X_train_encoded = encoder.fit_transform(X_train_raw)

xgb_model = XGBClassifier(random_state=42, 
                          n_jobs=-1,
                          learning_rate=0.1,
                          max_depth=6,
                          reg_alpha=1, # L1 regularization
                          n_estimators=100)

xgb_model = xgb_model.fit(X_train_encoded[features], y_train)


with open('model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('frequency_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
