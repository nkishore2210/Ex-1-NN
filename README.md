<H3>NAME: KISHORE N</H3>
<H3>REGISTER NO. 212222240049</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 29/02/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.

## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
df.isnull().sum()
df.duplicated()
df.describe()
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:

### DATASET:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/8d0886f5-2328-49b4-a94b-023b30b295a8)
### DROPPING THE UNWANTED DATASET:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/392df885-4680-41a7-a6f7-848ad3ce86c1)
### CHECKING NULL VALUES:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/a35fd620-7f99-4353-9d64-9986823a0f9c)
### CHECKING FOR DUPLICATION:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/b9e75543-472f-4a54-9c49-d2aca42c6156)
### DESCRIBING THE DATASET:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/bf28648f-5747-4d21-87f4-f8b312bb8703)
### SCALING THE DATASET:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/91a77bcb-bc95-4485-b8c1-b3a99e595951)
### X FEATURES:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/c8f68de2-52a6-477c-976b-4c8e121002d0)
### Y FEATURES:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/4c768cee-479a-4d3d-93ad-c53794911fb4)
### SPLITTING THE TRAINING AND TESTING DATASET:
![image](https://github.com/nkishore2210/Ex-1-NN/assets/118707090/335dd8a0-a072-47c8-95d2-79189463e913)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


