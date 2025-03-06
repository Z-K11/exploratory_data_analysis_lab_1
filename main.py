import pandas as pd 
import numpy as np 
import os 
import requests
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/iris_data.csv'
destination_folder='data'
#name of the destination folder 
os.makedirs(destination_folder,exist_ok=True)
# same as the makedir command in shell which creates a directory 
file_path=os.path.join(destination_folder,'iris_data.csv')
#creating filepath string by joining destination folder with the file name 
response = requests.get(url)
# sending an http get request to the file url server 
if os.path.exists(file_path):
    #checking if the file already exists or not 
    print(f'Data set already present at {file_path}')
    # if file already exists no need to re download the file 
else:
    with open(file_path,"wb") as file:
        #opening the file as a binary file and linking it to the file variable
        file.write(response.content)
        #writing the content of the http get response to the file 
    print(f'Download complete file saved at path {file_path}')

data_set = pd.read_csv(file_path)

print(f'number of rows in the data set {data_set.shape[0]}')
print(f'Data set column names {data_set.columns.tolist()}')
print(f'data type = {data_set.dtypes}')
print(data_set)
data_set['species']=data_set.species.str.replace('Iris-','')
# Removes the string Iris= from the species columns 
print(f'Count of species \n {data_set.species.value_counts()}')
stats_df = data_set.describe()
# 
print(stats_df)
# count prints the number of non null values in each column 
# mean prints the average value for each column 
# std prints the standar deviation a measure of how much values varry from the average 
# 25% prints the first quartile 25 percent of the data is below this value 
# 50% prints the median 
# 75% prints the third quartile 75% of the data is below this value \
# min prints the minimum value of each column
# max prints the maximum value of each column
stats_df.loc['range']=stats_df.loc['max']-stats_df.loc['min']
# loc is label based indexing and is inclusive of end 
new_fields= ['mean','25%','50%','75%','range']
# creating a list of new nable names 
stats_df = stats_df.loc[new_fields]
# re assigning the variable to have only the new labels 
stats_df.rename({'50%':'median'},inplace=True)
# renaming the 50% label as median
print(stats_df)