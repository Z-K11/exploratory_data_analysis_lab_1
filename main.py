import pandas as pd 
import numpy as np 
import os 
import requests
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/iris_data.csv'
destination_folder='data'
os.makedirs(destination_folder,exist_ok=True)
file_path=os.path.join(destination_folder,'iris_data.csv')
response = requests.get(url)
if os.path.exists(file_path):
    print(f'Data set already present at {file_path}')
else:
    with open(file_path,"wb") as file:
        file.write(response.content)
    print(f'Download complete file saved at path {file_path}')

data_set = pd.read_csv(file_path)
print(data_set.shape[0])
print(data_set.columns.tolist())