# Mengimpor library yang diperlukan
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer

# Import data yang akan diprediksi ke python
dataset_2 = pd.read_csv('test_loan.csv')

#cek jumlah null
dataset_2.isnull().sum()

#membagi data sesuai kategorinya(object/non-object)
cat_data_2 = []
num_data_2 = []

for i,c in enumerate(dataset_2.dtypes):
    if c == object:
        cat_data_2.append(dataset_2.iloc[:, i])
    else :
        num_data_2.append(dataset_2.iloc[:, i])

cat_data_2=pd.DataFrame(cat_data_2).transpose()
num_data_2=pd.DataFrame(num_data_2).transpose()

cat_data_2 = cat_data_2.apply(lambda x:x.fillna(x.value_counts().index[0]))
cat_data_2.isnull().sum().any() # no more missing data 

# Memproses data yang hilang (missing)
imputer = Imputer(missing_values= np.nan, strategy = 'mean', axis=0)
imputer.fit(num_data_2.values[:,(2,3)])
num_data_2.values[:,(2,3)] = imputer.transform(num_data_2.values[:,(2,3)])

imputer_mf=Imputer(missing_values=np.nan, strategy='most_frequent')
imputer_mf.fit(num_data_2.values[:, 4:5])
num_data_2.values[:,4:5]=imputer_mf.transform(num_data_2.values[:,4:5])

labelencoder = LabelEncoder()
cat_data_2.values[:, 2] = labelencoder.fit_transform(cat_data_2.values[:, 2])
cat_data_2.values[:, 4] = labelencoder.fit_transform(cat_data_2.values[:, 4])
cat_data_2.values[:, 5] = labelencoder.fit_transform(cat_data_2.values[:, 5])
cat_data_2.values[:, 6] = labelencoder.fit_transform(cat_data_2.values[:, 6])
cat_data_2.Dependents=cat_data_2.Dependents.replace({'3+':'3'})

#mendefinisikan variabel X_2(independen)
data_2 = pd.concat([cat_data_2, num_data_2], axis=1)
X_2=data_2.iloc[:, 2:12]


onehotencoder=OneHotEncoder(categorical_features=[4])
X1_2 = onehotencoder.fit_transform(X_2).toarray()
X_2=X1_2[:,1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_2 = sc.fit_transform(X_2)

#load model logistik regression
loaded_model = pickle.load(open("model_logreg.pkl","rb"))
result = loaded_model.predict(X_2)
result=pd.Series(result)

loan_values = { 1:'Y' , 0:'N'}

loan_status = result.map(loan_values)

dataset_2['Loan_status']=loan_status

dataset_2.to_csv("hasil_test_loan.csv")
