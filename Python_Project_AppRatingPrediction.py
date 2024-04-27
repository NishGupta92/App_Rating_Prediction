#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Ques. 1

# In[2]:


#Load the data file using pandas
df=pd.read_csv('googleplaystore.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# # Ques. 2

# In[6]:


df.isna().any()


# In[7]:


df.isna().sum()


# # Ques. 3

# In[8]:


df = df.dropna()


# In[9]:


df.isnull().any()


# In[10]:


df.shape


# # Ques. 4 - I

# In[11]:


df["Size"] = [ float(i.split('M')[0]) if 'M' in i else float(0) for i in df["Size"]  ]


# In[12]:


df.head()


# In[13]:


df ["Size"] = 1000*df["Size"]


# In[14]:


df


# In[15]:


df.info()


# # Ques. 4 - II

# In[16]:


df["Reviews"] = df["Reviews"].astype(float)


# In[17]:


df.info()


# # Ques. 4 - III

# In[18]:


df["Installs"] = [float(i.replace('+','').replace(',','')) if '+' in i or ',' in i else float(0) for i in df["Installs"]]


# In[19]:


df.head()


# In[20]:


df.info()


# In[21]:


df["Installs"] = df["Installs"].astype(int)


# In[22]:


df.info()


# # Ques. 4 - IV

# In[23]:


df["Price"] = [float(i.split('$')[1]) if '$' in i else float(0) for i in df["Price"]]


# In[24]:


df.head()


# In[25]:


df.info()


# In[26]:


df["Price"] = df["Price"].astype(int)


# In[27]:


df.info()


# # Ques. 4 - V(a)

# In[28]:


df.shape


# In[29]:


df.drop(df[(df["Reviews"] < 1) & (df["Reviews"] > 5 )].index, inplace = True)


# In[30]:


df.shape


# # Ques. 4 - V(b)

# In[31]:


df.shape


# In[32]:


df.drop(df[df["Installs"]<df["Reviews"]].index, inplace = True)


# In[33]:


df.shape


# # Ques. 4 - V(c)

# In[34]:


df.shape


# In[35]:


df.drop(df[(df["Type"]=="Free") & (df["Price"]>0)].index, inplace = True)


# In[36]:


df.shape


# # Ques. 5 - I

# In[37]:


sn.set(rc={'figure.figsize':(10,10)})


# In[38]:


sn.boxplot(x="Price",data= df);


# Yes, there are some outliers in the price column, there are some apps whose price is more than usual apps on the playstore.

# # Ques. 5 - II

# In[39]:


sn.boxplot(x="Reviews",data= df);


# Yes, there are some apps that have high no. of reviews.

# # Ques. 5 - III

# In[40]:


plt.hist(df["Rating"])


# There is a Negative skewness. Some apps seems to have higher ratings than usual.

# # Ques. 5 - IV

# In[41]:


plt.hist(df["Size"])


# There is a Positive skewness.

# # Ques. 6 - Handling Outliers

# # 6 - I

# In[42]:


df[df["Price"]>200].index.shape[0]


# In[43]:


df.drop(df[df["Price"]>200].index, inplace = True)


# In[44]:


df.shape


# # 6 - II

# In[45]:


df.drop(df[df["Reviews"]>2000000].index, inplace = True)


# In[46]:


df.shape


# # 6 - III

# In[47]:


#Find out the percentiles of Installs & decide a threshold as cutoff for outlier
df.quantile ([0.1,0.25,0.5,0.70,0.90,0.95,0.99], axis=0)


# In[48]:


df.drop(df[df["Installs"]>10000000].index, inplace = True)


# In[49]:


df.shape


# # Ques. 7 - Bivariate Analysis

# # 7 - I

# In[50]:


#Scatterplot/Joinplot for Rating Vs. Price
sn.scatterplot(x="Rating", y="Price", data=df)


# Paid apps are higher ratings as compare to free apps.

# # 7 - II

# In[51]:


#Scatterplot/Joinplot for Rating Vs. Size
sn.scatterplot(x="Rating", y="Size", data=df)


# Heavier apps are rated better because as the Size increases, the Ratings increases.

# # 7 - III

# In[52]:


#Scatterplot for Ratings Vs. Reviews
sn.scatterplot(x="Rating", y="Reviews", data=df)


# More Reviews mean better Ratings.

# # 7 - IV

# In[53]:


#Boxplot for Rating Vs. Content Rating
sn.boxplot(x="Rating", y="Content Rating", data=df)


# The above plot shows the apps for Everyone is worst rated as it contains the highest no. of outliers followed by apps for Mature 17+ & Everyone 10+ alongwith Teens. The Category Adults only 18+ is rated better & falls under most liked type.

# # 7 - V

# In[54]:


#Boxplot for Rating Vs. Category
sn.boxplot(x="Rating", y="Category", data=df)


# Event category has the best Ratings.

# # Ques. 8 - Data Preprocessing

# # 8 - I

# In[55]:


inp1=df.copy()


# In[56]:


inp1.head()


# In[57]:


inp1.skew()


# In[58]:


Reviews_skew=np.log1p(inp1["Reviews"])
inp1["Reviews"]=Reviews_skew


# In[59]:


Reviews_skew.skew()


# In[60]:


Installs_skew=np.log1p(inp1["Installs"])
inp1["Installs"]


# In[61]:


Installs_skew.skew()


# In[62]:


inp1.head()


# # 8 - II

# In[63]:


inp1.drop(["Last Updated","Current Ver","Android Ver","App","Type"], axis=1, inplace = True) #Had to drop Type column because it is string type and if I am not doing it then it is showing an error 


# In[64]:


inp1.head()


# In[65]:


inp1.shape


# # 8 - III

# In[66]:


inp2=inp1


# In[67]:


inp2.head()


# In[68]:


inp2["Category"].unique()


# In[69]:


inp2.Category=pd.Categorical(inp2.Category)

x=inp2[["Category"]]
del inp2["Category"]

dummies=pd.get_dummies(x, prefix="Category")
inp2=pd.concat([inp2, dummies], axis=1)
inp2.head()


# In[70]:


inp2["Genres"].unique()


# There are too many categories under Genres. Hence, we will try to reduce some categories which have very few samples under them & put them under one new common category i.e. "Other"

# In[71]:


#Create an empty list
lists=[]

#Get the total genres count & genres count of particular genres count less than 20, append those into the list
for i in inp2.Genres.value_counts().index:
    if inp2.Genres.value_counts()[i]<20:
        lists.append(i)

#Changing the genres which are in the list to other
inp2.Genres=["Other" if i in lists else i for i in inp2.Genres]


# In[72]:


inp2["Genres"].unique()


# In[73]:


#Storing the genres column into x variable & delete the genres column from dataframe inp2
inp2.Genres = pd.Categorical(inp2["Genres"])
x = inp2[["Genres"]]
del inp2["Genres"]
dummies = pd.get_dummies(x, prefix="Genres")
inp2 = pd.concat([inp2, dummies], axis=1)


# In[74]:


inp2.head()


# In[75]:


#Getting unique values in Column "Content Rating"
inp2["Content Rating"].unique()


# In[76]:


inp2["Content Rating"]=pd.Categorical(inp2["Content Rating"])

x=inp2[["Content Rating"]]
del inp2["Content Rating"]
dummies = pd.get_dummies(x, prefix="Content Rating")
inp2 = pd.concat([inp2, dummies], axis=1)


# In[77]:


inp2.head()


# In[78]:


inp2.shape


# # Ques. 9 and 10

# In[79]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics


# In[80]:


df2=inp2
X = df2.drop("Rating", axis=1)
Y = df2["Rating"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state = 0)


# # Ques. 11 - Model Building and Evaluation

# In[81]:


regressor=LinearRegression()
regressor.fit(X_train, Y_train)


# In[82]:


R2_score_train_data=round(regressor.score(X_train, Y_train),3)
print("The R2 value of the training set is : {}".format(R2_score_train_data)) 


# # Ques. 12 - Make Predictions on Test set and report R2

# In[83]:


Y_pred=regressor.predict(X_test)
R2_score_test_data=metrics.r2_score(Y_test,Y_pred)
R2_score_test_data


# In[84]:


R2_score_test_data=round(regressor.score(X_test,Y_test),3)
print("The R2 value of the training set is : {}".format(R2_score_test_data))


# In[ ]:




