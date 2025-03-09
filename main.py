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
print(f'Printing the mean exclusively \n {data_set.groupby('species').mean()}')
# grouping the data set by species and printing their mean 
print(f'printing the median exclusively \n {data_set.groupby('species').median()}')
print(f'Printing both the mean and median at the same time \n {data_set.groupby('species').agg(['mean','median'])}')
# Using the aggregate method you can call mutliple functions at the group same time 
print('Printing both the mean and median by explicityly calling functions ')
print(data_set.groupby('species').agg([np.mean,np.median]))
from pprint import pprint 
# importing pretty print library
agg_dict = {field: ['mean','median'] for field in data_set.columns if field != 'species'}
# using list comprehension to filter the list and itterate over all the elements in the column except 'species'
agg_dict['petal_length']='max'
pprint(agg_dict)
print(data_set.groupby('species').agg(agg_dict))
# The .agg method in pandas is designed to recognize certain strings as reference to common aggregation functions.
import matplotlib.pyplot as plt
ax = plt.axes()
# Creates a new set of axes which is a plotting area.
# ax is now an axes object , you can plot data in an axes object 
ax.scatter(data_set.sepal_length,data_set.sepal_width)
ax.set(xlabel='Sepal length (cm)',ylabel='Sepal width (cm)',title='Sepal length vs Sepal Width')
plots_directory='png_files'
os.makedirs(plots_directory,exist_ok=True)
sepal_length_vs_width_plot = os.path.join(plots_directory,'sepal_length_vs_width.png')
if os.path.exists(sepal_length_vs_width_plot):
    print(f'Plot already exists at {sepal_length_vs_width_plot}')
else:
    plt.savefig(os.path.join(plots_directory,'sepal_length_vs_width.png'))
plt.figure()
ax = plt.axes()
ax.hist(data_set.sepal_length,bins=25)
ax.set(xlabel='Sepal length',ylabel='Frequency',title='Distribution of Sepal Lengths')
histogram = 'sepal_length_histogram.png'
histogram_destination = os.path.join(plots_directory,histogram)
if os.path.exists(histogram_destination):
    print(f'Histogram exists at {histogram_destination}')
else: 
    plt.savefig(histogram_destination)
import seaborn as sns
plt.figure()
ax =plt.axes()
features=data_set[[x for x in data_set.columns if x !='species']]
feature = [x for x in data_set.columns if x !='species']
colors = ['blue','red','purple','green']
ax.hist(features,bins=25,alpha=0.5,color=colors,label=feature)
ax.set(xlabel='Size in cm ',ylabel='Frequency',title='Data Features Histogram')
ax.legend()
histogram='all_features.png'
histogram_destination = os.path.join(plots_directory,histogram)
if os.path.exists(histogram_destination):
    print(f'Histogram already exists at {histogram_destination}')
else:
    plt.savefig(histogram_destination)
sns.set_context('notebook')
ax = data_set.plot.hist(bins=25,alpha=0.5)
ax.set_label('Size (cm)')
seaborn_plot="overlaid_hist.png"
seaborn_path = os.path.join(plots_directory,seaborn_plot)
if os.path.exists(seaborn_path):
    print(f'plot already exists at path {seaborn_path}')
else:
    plt.savefig(seaborn_path)
fig, axes = plt.subplots(2,2,figsize=(10,10))
x=0
for i in range (2):
    for j in range(2):
        axes[i,j].hist(data_set[feature[x]],bins=25,color=colors[x],alpha=0.5)
        axes[i,j].set(xlabel='Size (cm)',ylabel='Frequency',title=feature[x])
        x+=1
four_plots = os.path.join(plots_directory,'tight_layout.png')
if os.path.exists(four_plots):
    print(f'Plot already exists at path {four_plots}')
else:
    plt.tight_layout()
    plt.savefig(four_plots)
data_set.boxplot(by='species')
plt.savefig(os.path.join(plots_directory,'boxplot.png'))
plot_data = (data_set
.set_index('species')
.stack().to_frame()
.reset_index()
.rename(columns={0:'Size','level_1':'Measurement'}))
print(plot_data.head())