# Mengimport library yang diperlukan
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer

# Import data ke python
dataset = pd.read_csv('train_loan.csv')

#cek jumlah data yang null
dataset.isnull().sum()

#membagi data sesuai kategorinya(object/non-object)
cat_data = []
num_data = []

for i,c in enumerate(dataset.dtypes):
    if c == object:
        cat_data.append(dataset.iloc[:, i])
    else :
        num_data.append(dataset.iloc[:, i])

cat_data=pd.DataFrame(cat_data).transpose()
num_data=pd.DataFrame(num_data).transpose()

# Memproses data yang hilang (missing)
cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

imputer = Imputer(missing_values= np.nan, strategy = 'mean')
imputer.fit(num_data.values[:,(2,3)])
num_data.values[:,(2,3)] = imputer.transform(num_data.values[:,(2,3)])

imputer_mf=Imputer(missing_values=np.nan, strategy='most_frequent')
imputer_mf.fit(num_data.values[:, 4:5])
num_data.values[:,4:5]=imputer_mf.transform(num_data.values[:,4:5])

#cek apakah masih ada data null
cat_data.isnull().sum().any() # no more missing data 
num_data.isnull().sum().any() # no more missing data 

#Encode data
labelencoder = LabelEncoder()
cat_data.values[:, 2] = labelencoder.fit_transform(cat_data.values[:, 2])
cat_data.values[:, 4] = labelencoder.fit_transform(cat_data.values[:, 4])
cat_data.values[:, 5] = labelencoder.fit_transform(cat_data.values[:, 5])
cat_data.values[:, 6] = labelencoder.fit_transform(cat_data.values[:, 6])
cat_data.Dependents=cat_data.Dependents.replace({'3+':'3'})

target_values = {'Y': 1 , 'N' : 0}

target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)

target = target.map(target_values) # 'Y' & 'N diganti menjadi 1 & 0

#mendefinisikan variabel X(independen) dan Y(dependen)
data = pd.concat([cat_data, num_data, target], axis=1)
X=data.iloc[:, 2:12]
y=data.iloc[:,12]

onehotencoder=OneHotEncoder(categorical_features=[4])
X1 = onehotencoder.fit_transform(X).toarray()
X=X1[:,1:]

# Membagi data ke dalam training dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

""" REGRESI LOGISTIK """
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state = 0)
classifier_logreg.fit(X_train, y_train)
 
# Memprediksi hasil modelnya ke test set
y_pred_logreg = classifier_logreg.predict(X_test)
 
# Membuat Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
as_logreg = accuracy_score(y_test, y_pred_logreg)

""" KNN """
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)
 
# Memprediksi Test set
y_pred_knn = classifier_knn.predict(X_test)
 
# Membuat Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
as_knn = accuracy_score(y_test, y_pred_knn)

""" SVM GAUSSIAN """
from sklearn.svm import SVC
classifier_svm_gaussian = SVC(kernel = 'rbf', random_state = 0)
classifier_svm_gaussian.fit(X_train, y_train)
 
# Memprediksi hasil test set
y_pred_svm_gaussian = classifier_svm_gaussian.predict(X_test)
 
# Membuat confusion matrix
cm_svm_gaussian = confusion_matrix(y_test, y_pred_svm_gaussian)
as_svm_gaussian = accuracy_score(y_test, y_pred_svm_gaussian)

""" SVM NO KERNEL """
from sklearn.svm import SVC
classifier_svm_nokernel = SVC(kernel = 'linear', random_state = 0)
classifier_svm_nokernel.fit(X_train, y_train)
 
# Memprediksi hasil test set
y_pred_svm_nokernel = classifier_svm_nokernel.predict(X_test)
 
# Membuat confusion matrix
cm_svm_nokernel = confusion_matrix(y_test, y_pred_svm_nokernel)
as_svm_nokernel = accuracy_score(y_test, y_pred_svm_nokernel)

""" NAIVE BAYES """
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
 
# Memprediksi hasil test set
y_pred_nb = classifier_nb.predict(X_test)
 
# Membuat confusion matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
as_nb = accuracy_score(y_test, y_pred_nb)

""" DECISION TREE CLASSIFICATION """
from sklearn.tree import DecisionTreeClassifier
classifier_dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dtc.fit(X_train, y_train)
 
# Memprediksi hasil test set
y_pred_dtc = classifier_dtc.predict(X_test)
 
# Membuat confusion matrix
cm_dtc = confusion_matrix(y_test, y_pred_dtc)
as_dtc = accuracy_score(y_test, y_pred_dtc)

""" RANDOM FOREST CLASSIFIER """
from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier_rfc.fit(X_train, y_train)
 
# Memprediksi hasil test set
y_pred_rfc = classifier_rfc.predict(X_test)
 
# Membuat confusion matrix
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
as_rfc = accuracy_score(y_test, y_pred_rfc)


score={'as_logreg':as_logreg, 'as_knn':as_knn, 'as_svm_gaussian':as_svm_gaussian, 'as_svm_nokernel':as_svm_nokernel, 'as_nb':as_nb, 'as_dtc':as_dtc, 'as_rfc':as_rfc}
score_list=[]
for i in score:
    score_list.append(score[i])
    u=max(score_list)
    if score[i]==u:
        v=i  
    print(f"{i}={score[i]}");   
print(f"The best method to use in this case is {v} with accuracy score {u}")

#Setelah tahu bahwa metode logistik regresi merupakan metode paling baik untuk case ini, maka kita dump model logistik regression
import pickle
pickle.dump(classifier_logreg, open("model_logreg.pkl","wb"))
