import pandas as pd


class Frequency_Transformer_Single():
    
    def __init__(self, name_f, name_Label):
        self.name_f = name_f
        self.Label = name_Label
        self.nf_name = name_f+"_ave"
        
    def fit(self, X, y=None):
        self.group = X.groupby(by = self.name_f)[self.Label].mean()
        self.med = np.nanmean(self.group.values)
 
        
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
        return self.Transform(X)
            
    def numeric_fill(self, X, y=None):
        for transformer in self.transformers:
            X = transformer.numeric_fill(X)
        return X
    
    def find_near(self, X, num):
        for transformer in self.transformers:
            X = transformer.find_near(X)
        return X
