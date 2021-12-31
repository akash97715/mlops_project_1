import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#for data preprocessing
from sklearn.decomposition import PCA
#for modeling
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import bios

my_dict = bios.read('config.yaml')
df = pd.read_csv(my_dict['Sorce_input_data_path_with_filename'])

DropCols = ['index', 'National Provider Identifier',
       'Last Name/Organization Name of the Provider',
       'First Name of the Provider', 'Middle Initial of the Provider','Street Address 1 of the Provider',
       'Street Address 2 of the Provider','Zip Code of the Provider',"HCPCS Code"]
df = df.drop(DropCols, axis = 1)
def RemoveComma(x):
    return x.replace(",","")
df["Average Medicare Allowed Amount"] = pd.to_numeric(df["Average Medicare Allowed Amount"].apply(lambda x: RemoveComma(x)),
                                                             errors= "ignore")
df["Average Submitted Charge Amount"] = pd.to_numeric(df["Average Submitted Charge Amount"].apply(lambda x: RemoveComma(x)),
                                                       errors = "ignore")
df["Average Medicare Payment Amount"] = pd.to_numeric(df["Average Medicare Payment Amount"].apply(lambda x: RemoveComma(x)),
                                                       errors = "ignore")
df["Average Medicare Standardized Amount"] = pd.to_numeric(df["Average Medicare Standardized Amount"].apply(lambda x: RemoveComma(x)),
                                                             errors = "ignore")
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

def RemoveComma(x):
    return x.replace(",","")
def Preprocessing(data):
    
    
    #1.Imputing Missing Values

    data["Credentials of the Provider"] = data["Credentials of the Provider"].fillna(data["Credentials of the Provider"].mode()[0])
    data["Gender of the Provider"] = data["Gender of the Provider"].fillna(data["Gender of the Provider"].mode()[0])
    

   #2.Binary Encoding.

    
    BEcols = [var for var in data.columns if data[var].dtype == "O"]
    print(BEcols)
    
    for col in BEcols:
        encoder = ce.BinaryEncoder(cols = [col])
        dfbin = encoder.fit_transform(data[col])
        data = pd.concat([data,dfbin], axis = 1)
        del data[col]

    #3. One-Hot-Encoding

#     data = pd.get_dummies(data,drop_first = True)
    
 
    #4. Standardization
 
    data_columns = data.columns
    std = StandardScaler()
    data = std.fit_transform(data)
    data = pd.DataFrame(data, columns = data_columns)
    
    return data
df = Preprocessing(df)
from sklearn.ensemble import IsolationForest

model = IsolationForest(n_estimators=my_dict['n_estimators'], max_samples=my_dict['max_samples'], 
                        contamination=my_dict['contamination'], max_features=my_dict['max_features'], bootstrap=my_dict['bootstrap'], n_jobs=my_dict['n_jobs'], 
                         verbose=my_dict['verbose'], warm_start=my_dict['warm_start'], random_state=my_dict['random_state'])
model.fit(df)

Y = model.predict(df)

Y[Y == 1] = 0
Y[Y == -1] = 1
df_shal = pd.read_csv(my_dict['Sorce_input_data_path_with_filename'])
df_shal['Anomaly_Flag']=Y
df_shal.to_csv(my_dict['Row_wise_anomaly_output_data_path'],index=False)