import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import os
import shutil

# Define the directory for FHE client/server files
fhe_directory = '/tmp/fhe_client_server_files/'

# Create the directory if it does not exist
if not os.path.exists(fhe_directory):
    os.makedirs(fhe_directory)
else:
    # If it exists, delete its contents
    shutil.rmtree(fhe_directory)
    os.makedirs(fhe_directory)

data=pd.read_csv('data/heart.xls')

data.info()   #checking the info

data_corr=data.corr()

plt.figure(figsize=(20,20))
sns.heatmap(data=data_corr,annot=True)
#Heatmap for data

feature_value=np.array(data_corr['output'])
for i in range(len(feature_value)):
    if feature_value[i]<0:
        feature_value[i]=-feature_value[i]

print(feature_value)

features_corr=pd.DataFrame(feature_value,index=data_corr['output'].index,columns=['correalation'])

feature_sorted=features_corr.sort_values(by=['correalation'],ascending=False)

feature_selected=feature_sorted.index

feature_selected     #selected features which are very much correalated

clean_data=data[feature_selected]

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier                  #using sklearn decisiontreeclassifier
from sklearn.model_selection import train_test_split

#making input and output dataset
X=clean_data.iloc[:,1:]
Y=clean_data['output']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)     #data is splited in traing and testing dataset

# feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#training our model
dt=XGBClassifier(max_depth=6)
dt.fit(x_train,y_train)
#dt.compile(x_trqin)

#predicting the value on testing data
y_pred=dt.predict(x_test)

#ploting the data
from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
accuracy=dt.score(x_test,y_test)
print("\nThe accuracy of decisiontreelassifier on Heart disease prediction dataset is "+str(round(accuracy*100,2))+"%")

joblib.dump(dt, 'heart_disease_dt_model.pkl')

from concrete.ml.sklearn.xgb import XGBClassifier as ConcreteXGBClassifier

fhe_compatible = ConcreteXGBClassifier.from_sklearn_model(dt, x_train, n_bits = 10)
fhe_compatible.compile(x_train)


#### server
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer

# Setup the development environment
dev = FHEModelDev(path_dir=fhe_directory, model=fhe_compatible)
dev.save()

# Setup the server
server = FHEModelServer(path_dir=fhe_directory)
server.load()







####### client

from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer

# Setup the client
client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()


# Load the dataset and select the relevant features
data = pd.read_csv('data/heart.xls')

# Perform the correlation analysis
data_corr = data.corr()

# Select features based on correlation with 'output'
feature_value = np.array(data_corr['output'])
for i in range(len(feature_value)):
    if feature_value[i] < 0:
        feature_value[i] = -feature_value[i]

features_corr = pd.DataFrame(feature_value, index=data_corr['output'].index, columns=['correlation'])
feature_sorted = features_corr.sort_values(by=['correlation'], ascending=False)
feature_selected = feature_sorted.index

# Clean the data by selecting the most correlated features
clean_data = data[feature_selected]

# Extract the first row of feature data for prediction (excluding 'output' column)
sample_data = clean_data.iloc[0, 1:].values.reshape(1, -1)  # Reshape to 2D array for model input

encrypted_data = client.quantize_encrypt_serialize(sample_data)



##### end client

encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)

result = client.deserialize_decrypt_dequantize(encrypted_result)
print(result)