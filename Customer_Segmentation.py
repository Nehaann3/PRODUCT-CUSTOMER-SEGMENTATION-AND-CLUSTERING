#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv("C:/Users/MY BOOK/Downloads/Mall_Customers.csv")


# In[5]:


df.head()


# In[6]:


df.describe()


# In[45]:


sns.distplot(df['Annual Income (k$)'])


# In[46]:


columns = ['Age','Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[47]:


columns = ['Age','Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(data=df,x=i, fill=True,hue='Gender')


# In[10]:


columns = ['Age','Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender', y=df[i])


# In[11]:


sns.scatterplot(data=df,x= 'Annual Income (k$)',y= 'Spending Score (1-100)')


# In[48]:


sns.pairplot(df,hue = 'Gender')


# In[13]:


df.groupby(['Gender'])[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[14]:


numeric_df = df.select_dtypes(include=[float, int])
numeric_df.corr()


# In[16]:


clustering1 = KMeans(n_clusters = 6)


# In[49]:


clustering1.fit(df[['Annual Income (k$)']])


# In[50]:


clustering1.labels_


# In[19]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[20]:


df['Income Cluster'].value_counts()


# In[21]:


clustering1.inertia_


# In[51]:


inertia_scores = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[23]:


inertia_scores


# In[24]:


plt.plot(range(1,11),inertia_scores)


# In[25]:


df.groupby('Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[52]:


clustering2 = KMeans()
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[53]:


inertia_scores2 = []
for i in range(1,11):
    kmeans2 = KMeans(n_clusters = i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),inertia_scores2)


# In[28]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[42]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y= centers['y'],s=100,c='black', marker ='*')

sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)',hue = 'Spending and Income Cluster',palette = 'tab10')
plt.savefig('clustering_bivariate.png')


# In[43]:


df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


scaler = StandardScaler()


# In[33]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[34]:


dff.columns


# In[35]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)',
       'Income Cluster', 'Spending and Income Cluster', 'Gender_Male']]


# In[36]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
dff = scale.fit_transform(dff)


# In[37]:


dff = pd.DataFrame(scale.fit_transform(dff)) 


# In[54]:


inertia_scores3 = []
for i in range(1,11):
    kmeans3 = KMeans(n_clusters = i)
    kmeans3.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),inertia_scores3)


# In[39]:


df


# In[40]:


df.to_csv('Segmentation and Clustering.csv')


# In[ ]:




