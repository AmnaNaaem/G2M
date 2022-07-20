#!/usr/bin/env python
# coding: utf-8

# Calling Important Libraries:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Calling all four Data files:

# Calling data sets as follows:
# 1. Cab data set as df1.
# 2. City data set as df2.
# 3. Customer ID as df3.
# 4. Transaction ID as df4.

# In[2]:


data1=pd.read_csv("Cab_Data.csv")
df1=data1.copy()


# In[3]:


df1.head()


# In[4]:


data2=pd.read_csv("City.csv")
df2=data2.copy()


# In[5]:


df2.head()


# In[6]:


data3=pd.read_csv("Customer_ID.csv")
df3=data3.copy()


# In[7]:


df3.head()


# In[8]:


data4=pd.read_csv("Transaction_ID.csv")
df4=data4.copy()


# In[9]:


df4.head()


# In[10]:


df1.columns


# In[11]:


df2.columns


# In[12]:


df3.columns


# In[13]:


df4.columns


# # Merging Data sets using Common information:

# In[14]:


# data set 1 and data set 4 on the basis of Transaction ID:
data=df1.merge(df4,on='Transaction ID')


# In[15]:


data.head()


# In[16]:


# Combining data set 3 with data using customer ID:
data=data.merge(df3,on='Customer ID')


# In[17]:


#Now combining data set 2 using City information and copying data into name df for further analysis:
data=data.merge(df2,on='City')
df=data.copy()


# In[18]:


df.head()


# In[19]:


df.tail()


# # Basic Analysis 

# In[20]:


# Command to show number of columns and rows:
df.shape


# In[21]:


# Command to see the column names:
df.columns


# In[22]:


# Now we will analyse the Data type of all variables and change the type if needed:
df.dtypes


# # Converting data type of Variable Transaction ID, Customer ID, Users, Population and Date of Travel:

# Coverting Transaction ID and Customer ID as object type.
# Converting Users and Population as Integer type.
# Converting Date of Travel as date/time.

# In[23]:


df['Transaction ID']=df['Transaction ID'].astype(object)


# In[24]:


df['Customer ID']=df['Customer ID'].astype(object)


# In[25]:


df['Users'] = df['Users'].str.replace(',', '')


# In[26]:


df['Users']=pd.to_numeric(df['Users'])


# In[27]:


df['Population'] = df['Population'].str.replace(',', '')


# In[28]:


df['Population'] = pd.to_numeric(df['Population'])


# In[29]:


df['Date of Travel']=pd.to_datetime(df['Date of Travel'])


# In[30]:


df.dtypes


# # Summary Statistic of Data:

# In[31]:


# This command will give us summary statistics for interger and float type variables:
df.describe()


# In[32]:


# This command will provide us information regarding object type variables in the data:
df.describe(include='O')


# In[33]:


df_sum = df.select_dtypes(include = ['float64', 'int64'])
df_sum.head()


# # Missing Value Analysis:

# In[34]:


df.isnull().sum()


# # Duplicate Value Analysis:

# In[35]:


df.duplicated().sum()


# # Outlier Detection for features with Integer and Float type:

# In[36]:


outliers=[]
# For each feature find the data points with extreme high or low values
for feature in df_sum.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(df_sum[feature],25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(df_sum[feature],75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3-Q1) 
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    out=df[~((df_sum[feature] >= Q1 - step) & (df_sum[feature] <= Q3 + step))]
    display(out)
    outliers=outliers+list(out.index.values)
    
# Select the indices for data points you wish to remove
outliers = list(set([x for x in outliers if outliers.count(x) > 1]))    
print ("Outliers: {}".format(outliers))


# In[ ]:





# In[37]:


Profit=df['Price Charged']-df['Cost of Trip']
df['Profit']=Profit


# # Exploration of Complete and Important Variables:

# Company and Gender Preference: 

# In[38]:


df['Company'].unique()


# In[39]:


df.groupby('Company')['Gender'].count()


# In[40]:


#plt.figure(figsize=(15,6))
sns.countplot(x='Company', hue='Gender', data=df);


# City and Company

# In[41]:


df['City'].unique()


# In[42]:


plt.figure(figsize=(15,6))
sns.factorplot(x='Company', col='City', kind='count',data=df);


# Company, City and Gender:

# In[43]:


sns.set_context('poster')
sns.catplot(x='Company', col='City', hue='Gender', data=df, kind='count');


# Company and Price Charged:

# In[44]:


df.groupby(["Company"]).agg({"Price Charged" : "mean"}).sort_values(by="Price Charged", ascending=False)


# In[45]:


x = [458.181990,310.800856]
labels = ['Yellow Cab','Pink Cab']
colors = ['tab:blue','tab:red']

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors, autopct='%.0f%%')
ax.set_title('Mean Price Charged')
plt.show()


# Company and Cost of Trip:

# In[46]:


df.groupby(["Company"]).agg({"Cost of Trip" : "mean"}).sort_values(by="Cost of Trip", ascending=False)


# In[47]:


x = [297.922004,248.148682]
labels = ['Yellow Cab','Pink Cab']
colors = ['tab:blue','tab:red']

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors, autopct='%.0f%%')
ax.set_title('Mean Cost of Trip')
plt.show()


# In[48]:


df.groupby('Company')['Profit'].mean()


# In[49]:


df.groupby(["Company"]).agg({"Profit" : "sum"})[["Profit"]].apply(lambda x: 100*x/x.sum()).sort_values(by="Profit", ascending=False)


# In[50]:


x = [89.240674,10.759326]
labels = ['Yellow Cab','Pink Cab']
colors = ['tab:blue','tab:red']

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors, autopct='%.0f%%')
ax.set_title('Profit')
plt.show()


# Company, City and Users: 

# In[51]:


temp=df.groupby(['Company', 'City'])['Users'].count().reset_index(name='total')


# In[52]:


temp


# In[53]:


plt.figure(figsize=(15,6));
sns.barplot(data=temp, y='City', x='total', hue='Company', dodge=True);


# In[54]:


df.groupby('Company')['Users'].sum()


# In[55]:


df.groupby(["Company"]).agg({"Users" : "sum"})[["Users"]].apply(lambda x: 100*x/x.sum()).sort_values(by="Users", ascending=False)


# Age and Company:

# In[56]:


df['Age'].unique()


# Defining Age Categories:

# In[57]:


df['AgeCat']=''
df.loc[ df['Age'] <= 20, 'AgeCat'] = '0-20'
df.loc[(df['Age'] > 20) & (df['Age'] <= 40), 'AgeCat'] = '21-40'
df.loc[(df['Age'] > 40) & (df['Age'] <= 60), 'AgeCat'] = '41-60'
df.loc[ df['Age'] > 60, 'AgeCat']= '60+'


# In[58]:


df.head()


# In[59]:


age_cab=df.groupby(['Company','AgeCat']).count()['Users']


# In[60]:


age_cab


# In[61]:


df.groupby(['Company','AgeCat']).count()['Users'].plot(kind='bar',stacked=False)


# In[62]:


df.groupby(["AgeCat","Company"]).agg({"Users" : "sum"})[["Users"]].apply(lambda x: 100*x/x.sum()).sort_values(by="Users", ascending=False).plot(kind='bar')


# In[63]:


df.groupby(["AgeCat"]).agg({"Users" : "sum"})[["Users"]].apply(lambda x: 100*x/x.sum()).sort_values(by="Users", ascending=False).plot(kind='bar')


# In[64]:


df.groupby(['Company', 'AgeCat','City']).count()['Users']


# In[ ]:





# # Time Analysis:

# In[65]:


df['months'] = pd.DatetimeIndex(df['Date of Travel']).month
df['day'] = pd.DatetimeIndex(df['Date of Travel']).day
df['year'] = pd.DatetimeIndex(df['Date of Travel']).year
df.head()


# In[66]:


years = df['year'].unique()


# In[67]:


fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='Profit', data=df, ax=axes[0])
sns.boxplot(x='months', y='Profit', data=df.loc[~df.year.isin([2016, 2018]), :])


# In[68]:


fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='Users', data=df, ax=axes[0])
sns.boxplot(x='months', y='Users', data=df.loc[~df.year.isin([2016, 2018]), :])


# In[69]:


fig, axes = plt.subplots(1, 1, figsize=(20,7), dpi= 80)
sns.boxplot(x='months', y='Company', data=df.loc[~df.year.isin([2016, 2018]), :])


# In[70]:


w=df.groupby(['year','Users'])['Profit'].mean()


# In[71]:


w.plot(figsize=(15, 6))
plt.show()


# In[72]:


z=df.groupby(['year','Company'])['Profit'].mean()


# In[73]:


z.plot(kind='bar')
plt.show()


# # Feature Selection

# In[74]:


from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as preprocessing 


# In[75]:


def number_encode_features(df): 
    result = df.copy()     
    encoders = {}     
    for column in result.columns:         
        if result.dtypes[column] == np.object:             
            encoders[column] = preprocessing.LabelEncoder() 
            result[column] = encoders[column].fit_transform(result[column]
) 
    return result, encoders  
# Calculate the correlation and plot it 
encoded_data, _ = number_encode_features(df)
encoded_data.drop(['Transaction ID','Customer ID'],axis=1).corr()


# In[76]:


df=df.drop(['Users','Age','Price Charged','Cost of Trip','months','day','year'],axis=1)


# In[77]:


df.head()


# In[78]:


df1 = df.groupby('Date of Travel')['Profit'].sum().reset_index()


# In[79]:


df1 = df1.set_index('Date of Travel')
df1.index


# In[80]:


y = df1['Profit'].resample('MS').mean()


# In[81]:


y.plot(figsize=(15, 6))
plt.show()


# In[82]:


import itertools


# In[83]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[84]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[85]:


import statsmodels.api as sm


# In[86]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[87]:


pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2016':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Profit')
plt.legend()
plt.show()


# In[88]:


y_forecasted = pred.predicted_mean
y_truth = y['2018-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[89]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# In[90]:


pred_uc = results.get_forecast(steps=24)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Profit')
plt.legend()
plt.show()


# # Hypothesis Testing:

# In[91]:


from scipy import stats


# <h1><center>Anova Test!</center></h1>

# In[92]:


stats.f_oneway(df['Profit'][df['Company'] == 'Pink Cab'],
               df['Profit'][df['Company'] == 'Yellow Cab'])

