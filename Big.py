import pandas as pd
import numpy as np
data=pd.read_csv('OnlineRetail.csv',encoding = "ISO-8859-1")
print(data)
print(data.describe())
print(data.info())
print(data.columns)
print(data.shape)
print(data.isnull().sum())
df_null=(round(100*(data.isnull().sum())/len(data),2))
print(df_null)

#print(data['InvoiceDate'])
#data =data.drop('StockCode',axis=1)
#print(data.shape)
#changing custonerId to string
data['CustomerID'] = data['CustomerID'].astype(str)
print(data.info())
#Data preparation
#coming up with new column
data['Amount'] = data['Quantity']*data['UnitPrice']
print(data.head())
print(data.info())
data_monetary = data.groupby('CustomerID')['Amount'].sum()
print(data_monetary.head())
#most sold product
Most_sold_product =data.groupby('Description')['Quantity'].sum()
Most_sold_product=Most_sold_product.sort_values(ascending=False).head(1)
print(Most_sold_product)
#region
Country =data.groupby('Country')['Quantity'].sum()
most_sold_country=Country.sort_values(ascending=False).head(1)
print(most_sold_country)
#FREQUENCY
frequency_sold =data.groupby('Description')['InvoiceNo'].count()
frequency_sold=frequency_sold.sort_values(ascending=False).head(1)
print(frequency_sold)
#sum of last total sales of last month of the year
#convert datetime to proper datatype
data['InvoiceDate']= pd.to_datetime(data['InvoiceDate'],format='%m/%d/%Y %H:%M')
print(data.head())
print(data.info())
#compute the max time to compute the last transtion date
max_date =max(data['InvoiceDate'])
print(max_date)
min_date =min(data['InvoiceDate'])
print(min_date)
#total number of days
diff=(max_date-min_date)
print(diff)

#total sales for the last 30 days
start_date =max_date-pd.DateOffset(days=30)
print(start_date)
data['Totalsales']=data['UnitPrice']*data['Quantity']
print(data.info())
total_sales=data.groupby('InvoiceDate')['Totalsales'].sum()
total_sales=total_sales.sum()
print(total_sales)
total_Amount_sales=data['Amount'].count()
print(total_Amount_sales)

#KMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#define input data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Amount']])
x = scaled_data 
x=data[['Amount']]

wcss = []
model={}
silhouette_scores = []
for k in range(2,8):
  kmeans = KMeans(n_clusters=k,init='random',random_state=0).fit(x)
  wcss.append(kmeans.inertia_)
  silhouette_scores.append(silhouette_score(x, kmeans.labels_))
  
#plot the
plt.plot(range(2,8),wcss, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for K-Means Clustering')
plt.show()


# Step 4: Plot silhouette scores
plt.plot(range(2,8), silhouette_scores, marker='s',label='Silhouette Score')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for K-Means Clustering')
plt.show()
print("Code execution complete!")




