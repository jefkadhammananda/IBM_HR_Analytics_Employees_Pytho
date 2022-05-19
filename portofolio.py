#!/usr/bin/env python
# coding: utf-8

# #Dataset : IBM HR Analytics Employees
# https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
# 
# Attrition = (Resign / Karyawan yang tidak ada dalam perusahaan)
# 
# - Masalah:
# Menurunnya peforma perusahaan karena adanya kekosongan waktu jabatan yang disebabkan oleh ketidakpastian karyawan apakah resign atau tidak (Attrition).
# 
# - Tujuan:
# Mengetahui karyawan attrition.
# 
# Metode:
# Classifaction : SVM, Logistic Regression dan KNN

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Read dataset
df = pd.read_csv('IBM HR.csv')
df.head()


# In[3]:


#Cek apa ada data yang kosong
df.info()


# In[4]:


#Tunjukkan total data baris, kolum
df.shape


# In[5]:


#Tujukkan features / attribute / columns
df.columns


# In[6]:


# Tunjukkan tiap tipe attribute dari dataset
df.dtypes


# In[7]:


#Mencoba mencari dan melihat attribute dalam tipe kategori
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
print("Jumlah attribute tipe kategori:",len(categorical_features))
categorical_features


# In[8]:


#Mencoba mencari dan melihat attribute dalam tipe angka/numerik
numerical_features = [feature for feature in df.columns if feature not in categorical_features]
print("Jumlah attribute tipe angka:",len(numerical_features))
print(numerical_features)


# In[9]:


#Melihat jumlah pembagian data pada tiap kategori
for i in categorical_features:
    print(df[[i]].value_counts(), "\n")


# In[10]:


#Mencari nomor unik pada tiap numberik attribute
for i in numerical_features:
    print(i, df[i].nunique())


# # Attribute yang di drop
# ['Over18',
# "BusinessTravel",
# 'DailyRate',
# 'EmployeeCount',
# 'EmployeeNumber',
# 'HourlyRate',
# 'MonthlyRate',
# 'StandardHours',
# 'StockOptionLevel']

# In[11]:


df_filter = df.drop(['Over18',"BusinessTravel",'DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate','StandardHours','StockOptionLevel'], axis = 1)
df_filter


# In[12]:


#Mencoba mencari dan melihat attribute dalam tipe kategori
categorical_features = [feature for feature in df_filter.columns if df_filter[feature].dtype == 'O']
print("Jumlah attribute tipe kategori:",len(categorical_features))
categorical_features


# In[13]:


for i in categorical_features:
    print(df_filter[[i]].value_counts(), "\n")


# In[14]:


#Mencoba mencari dan melihat attribute dalam tipe angka/numerik
numerical_features = [feature for feature in df_filter.columns if feature not in categorical_features]
print("Jumlah attribute tipe angka:",len(numerical_features))
print(numerical_features)


# In[15]:


#Mencari nomor unik pada tiap numberik attribute
for i in numerical_features:
    print(i, df_filter[i].nunique())


# In[16]:


# Membuat list untuk attribute numerik, keunikan tidak melebihi 11 macam
discrete_numerical_features = []
for i in numerical_features:
    if (df_filter[i].nunique()<11):
        discrete_numerical_features.append(i)


# In[18]:


#Melihat summary statistik data numerik
df_filter.describe()


# In[19]:


#Melihat summary data kategorikal
df_filter.describe(include = "O")


# #Visualisasi

# In[20]:


#Mapping ubah yes jadi angka 1 dan no jadi angka 0 pada attribute attrition
#Attrition_mapping = {"Yes": 1, "No": 0}
#df_filter['Attrition'] = df_filter['Attrition'].map(Attrition_mapping)


# In[21]:


#Melihat perbandingan pada attribute attrition
Attrition_mapping = {"Yes": 1, "No": 0}
df_filter['Attrition'].map(Attrition_mapping)
sns.countplot(df_filter['Attrition'])


# In[22]:


#Menghitung karyawan yang attrition
attrition = df_filter[(df_filter['Attrition'] == "Yes")]
no_attrition = df_filter[(df_filter['Attrition'] == "No")]
print('Percentage of Attrition: {}'.format(len(attrition)/len(df_filter)))
print('Percentage of No Attrition: {}'.format(len(no_attrition)/len(df_filter)))


# In[23]:


df_filter.groupby(["Gender","Attrition"]).size().unstack()


# In[24]:


df_filter.groupby(["Gender","Attrition"]).size().unstack().plot(kind="bar")
plt.show()


# In[25]:


df_filter.groupby(["Department","Attrition"]).size().unstack()


# In[26]:


df_filter.groupby(["Department","Attrition"]).size().unstack().plot(kind="bar")
plt.show()


# In[80]:


df_filter.groupby(["EducationField","Attrition"]).size().unstack()


# In[28]:


df_filter.groupby(["EducationField","Attrition"]).size().unstack().plot(kind="bar")
plt.show()


# In[81]:


df_filter.groupby(["MaritalStatus","Attrition"]).size().unstack()


# In[30]:


df_filter.groupby(["MaritalStatus","Attrition"]).size().unstack().plot(kind="bar")
plt.show()


# In[31]:


df_filter.groupby(["OverTime","Attrition"]).size().unstack()


# In[32]:


df_filter.groupby(["OverTime","Attrition"]).size().unstack().plot(kind="bar")
plt.show()


# In[33]:


sns.set_style('whitegrid')
sns.distplot(df_filter['Age'], bins = 10)


# In[34]:


g = sns.FacetGrid(df_filter, col='Attrition')
g.map(plt.hist, 'Age', bins=15)


# In[35]:


g = sns.FacetGrid(df_filter, col='Attrition')
g.map(plt.hist, 'MonthlyIncome', bins=15)


# In[36]:


g = sns.FacetGrid(df_filter, col='Attrition')
g.map(plt.hist, 'DistanceFromHome', bins=15)


# In[37]:


df_filter.groupby(["PerformanceRating","Attrition"]).size().unstack()


# In[38]:


df_filter.groupby(["PerformanceRating","Attrition"]).size().unstack().plot(kind="bar")
plt.show()


# In[39]:


plt.figure(figsize = (12,6))
sns.countplot(df['TotalWorkingYears'], hue = df_filter['Attrition'])
plt.show()


# #Preprocessing Data

# In[40]:


#Menunjukkan data kolerasi dalam heatmap
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
k_corr_matrix1 =df_filter.corr()
plt.figure(figsize=(20,14))
sns.heatmap(k_corr_matrix1, annot=True, cmap=plt.cm.RdBu_r)
plt.title('Heatmap for Correlation between Features')


# In[41]:


# Membuat fungsi untuk mencari attribute dengan tingkat kolerasi tertentu

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[42]:


tingkat_kolerasi = 0.7
corr_features = correlation(df_filter, tingkat_kolerasi)
print("Terdapat", len(set(corr_features)), "Macam Kolerasi dengan tingkat kolerasi", tingkat_kolerasi)


# In[43]:


corr_features


# In[44]:


categorical_features = [feature for feature in df_filter.columns if df_filter[feature].dtype == 'O']
categorical_features


# In[45]:


#Label-Encoding ordinal categorical features 
from sklearn.preprocessing import LabelEncoder
for c in categorical_features:
    lbl = LabelEncoder() 
    lbl.fit(list(df_filter[c].values)) 
    df_filter[c] = lbl.transform(list(df_filter[c].values))


# In[46]:


df_filter.head()


# In[47]:


# Attribute yang dipakai
df_filter.columns


# In[48]:


df_filter.shape


# #Split Train & Test Data

# In[49]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = np.asarray(df_filter.drop(['Attrition'], axis = 1))
X = preprocessing.StandardScaler().fit(X).transform(X)
y = np.asarray(df_filter['Attrition'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# In[50]:


print('Attribute faktor dalam bentuk array : \n',X[0:5])
print('\nAttribute target dalam bentuk array : \n',y[0:5])
print ('\nTrain set:', X_train.shape,  y_train.shape)
print ('\nTest set:', X_test.shape,  y_test.shape)


# #Model SVM

# In[51]:


from sklearn import svm
model_svm = svm.SVC(kernel='rbf')
model_svm.fit(X_train, y_train) 


# In[52]:


yhat = model_svm.predict(X_test)
yhat [0:5]


# In[53]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[54]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[55]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

print("Train set Accuracy: ", jaccard_similarity_score(y_train, model_svm.predict(X_train)))
print("Test set Accuracy: ", jaccard_similarity_score(y_test, yhat))


# In[56]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 

print("Train set Accuracy: ", f1_score(y_train, model_svm.predict(X_train), average='weighted'))
print("Test set Accuracy: ", f1_score(y_test, yhat, average='weighted'))


# In[57]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Attrition(0)="No"','Attrition(1)="Yes"'],normalize= False,  title='Confusion matrix')


# In[58]:


print (classification_report(y_test, yhat))


# #Model Logistic Regression

# In[59]:


X = np.asarray(df_filter.drop(['Attrition'], axis = 1))
y = np.asarray(df_filter['Attrition'])


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Attribute faktor dalam bentuk array : \n',X[0:5])
print('\nAttribute target dalam bentuk array : \n',y[0:5])
print ('\nTrain set:', X_train.shape,  y_train.shape)
print ('\nTest set:', X_test.shape,  y_test.shape)


# In[62]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[63]:


yhat = LR.predict(X_test)
yhat


# In[64]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob[0:30]


# In[65]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

print("Train set Accuracy: ", jaccard_similarity_score(y_train, LR.predict(X_train)))
print("Test set Accuracy: ", jaccard_similarity_score(y_test, yhat))


# In[66]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 

print("Train set Accuracy: ", f1_score(y_train, LR.predict(X_train), average='weighted'))
print("Test set Accuracy: ", f1_score(y_test, yhat, average='weighted') )


# In[68]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Attrition(0)="No"','Attrition(1)="Yes"'],normalize= False,  title='Confusion matrix')


# In[69]:


print (classification_report(y_test, yhat))


# #Model KNN (K_Nearest_Neighbors)

# In[70]:


from sklearn.neighbors import KNeighborsClassifier
X = np.asarray(df_filter.drop(['Attrition'], axis = 1))
y = np.asarray(df_filter['Attrition'])


# In[71]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Attribute faktor dalam bentuk array : \n',X[0:5])
print('\nAttribute target dalam bentuk array : \n',y[0:5])
print ('\nTrain set:', X_train.shape,  y_train.shape)
print ('\nTest set:', X_test.shape,  y_test.shape)


# In[73]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[74]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[75]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[76]:


Ks = 50
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[77]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[78]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[79]:


k = 7
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[ ]:




