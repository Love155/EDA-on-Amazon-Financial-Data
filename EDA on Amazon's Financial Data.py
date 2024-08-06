#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# # Load Data

# In[3]:


df = pd.read_json(r"D:\financial_data.json")
df.head()


# # Check & Understand the data by info & describing

# In[5]:


df.info()


# # Extract Data as per requriment

# In[10]:


x = df.iloc[:,[0,1,2,4,5,6,7,11,12,14,15,16,17,18,24,30]]
x


# In[15]:


x.describe().T


# # Check for missing values

# In[11]:


plt.figure(figsize=(10, 6))

sns.heatmap(x.isnull(), cbar=False, cmap='viridis')

plt.show()


# In[13]:


x.isnull().sum()


# # Imputing data

# In[23]:


x['Date'].fillna(method = 'ffill', inplace=True)
x['Open'].fillna(x['Open'].median(), inplace=True)
x['High'].fillna(x['High'].median(), inplace=True)
x['Close'].fillna(x['Close'].median(), inplace=True)
x['Volume'].fillna(x['Volume'].median(), inplace=True)
x['Adj_Close'].fillna(x['Adj_Close'].median(), inplace=True)
x['Market_Cap'].fillna(x['Market_Cap'].median(), inplace=True)
x['Revenue'].fillna(x['Revenue'].median(), inplace=True)
x['Gross_Profit'].fillna(x['Gross_Profit'].median(), inplace=True)
x['Net_Income'].fillna(x['Net_Income'].mean(), inplace=True)
x['Cash_Flow'].fillna(x['Cash_Flow'].mean(), inplace=True)
x['Assets'].fillna(x['Assets'].median(), inplace=True)
x['Liabilities'].fillna(x['Liabilities'].median(), inplace=True)
x['Equity'].fillna(x['Equity'].median(), inplace=True)
x['Current_Ratio'].fillna(x['Current_Ratio'].median(), inplace=True)
x['Shares_Outstanding'].fillna(x['Shares_Outstanding'].median(), inplace=True)


# In[24]:


x.isnull().sum()


# # Detecting Outliers

# In[27]:


plt.figure(figsize=(25,10))

sns.boxplot(x)

plt.show()


# In[29]:


sns.boxplot(x['Open'])


# In[32]:


Q1 = x['Open'].quantile(0.25)

Q3 = x['Open'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Open'] = np.where(x['Open'] > upper_bound, upper_bound, x['Open'])

x['Open'] = np.where(x['Open'] < lower_bound, lower_bound, x['Open'])

sns.boxplot(x['Open'])

plt.title("Identify Outliers in Open")

plt.show()


# In[34]:


sns.boxplot(x['High'])


# In[38]:


Q1 = x['High'].quantile(0.25)

Q3 = x['High'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['High'] = np.where(x['High'] > upper_bound, upper_bound, x['High'])

x['High'] = np.where(x['High'] < lower_bound, lower_bound, x['High'])

sns.boxplot(x['High'])

plt.title("Identify Outliers in High")

plt.show()


# In[37]:


sns.boxplot(x['Close'])


# In[39]:


Q1 = x['Close'].quantile(0.25)

Q3 = x['Close'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Close'] = np.where(x['Close'] > upper_bound, upper_bound, x['Close'])

x['Close'] = np.where(x['Close'] < lower_bound, lower_bound, x['Close'])

sns.boxplot(x['Close'])

plt.title("Identify Outliers in Close")

plt.show()


# In[41]:


sns.boxplot(x['Volume'])


# In[42]:


sns.boxplot(x['Adj_Close'])


# In[43]:


Q1 = x['Adj_Close'].quantile(0.25)

Q3 = x['Adj_Close'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Adj_Close'] = np.where(x['Adj_Close'] > upper_bound, upper_bound, x['Adj_Close'])

x['Adj_Close'] = np.where(x['Adj_Close'] < lower_bound, lower_bound, x['Adj_Close'])

sns.boxplot(x['Adj_Close'])

plt.title("Identify Outliers in Adj_Close")

plt.show()


# In[44]:


sns.boxplot(x['Market_Cap'])


# In[45]:


Q1 = x['Market_Cap'].quantile(0.25)

Q3 = x['Market_Cap'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Market_Cap'] = np.where(x['Market_Cap'] > upper_bound, upper_bound, x['Market_Cap'])

x['Market_Cap'] = np.where(x['Market_Cap'] < lower_bound, lower_bound, x['Market_Cap'])

sns.boxplot(x['Market_Cap'])

plt.title("Identify Outliers in Market_Cap")

plt.show()


# In[46]:


sns.boxplot(x['Revenue'])


# In[48]:


Q1 = x['Revenue'].quantile(0.25)

Q3 = x['Revenue'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Revenue'] = np.where(x['Revenue'] > upper_bound, upper_bound, x['Revenue'])

x['Revenue'] = np.where(x['Revenue'] < lower_bound, lower_bound, x['Revenue'])

sns.boxplot(x['Revenue'])

plt.title("Identify Outliers in Revenue")

plt.show()


# In[49]:


sns.boxplot(x['Gross_Profit'])


# In[50]:


Q1 = x['Gross_Profit'].quantile(0.25)

Q3 = x['Gross_Profit'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Gross_Profit'] = np.where(x['Gross_Profit'] > upper_bound, upper_bound, x['Gross_Profit'])

x['Gross_Profit'] = np.where(x['Gross_Profit'] < lower_bound, lower_bound, x['Gross_Profit'])

sns.boxplot(x['Gross_Profit'])

plt.title("Identify Outliers in Gross_Profit")

plt.show()


# In[52]:


sns.boxplot(x['Net_Income'])


# In[53]:


Q1 = x['Net_Income'].quantile(0.25)

Q3 = x['Net_Income'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Net_Income'] = np.where(x['Net_Income'] > upper_bound, upper_bound, x['Net_Income'])

x['Net_Income'] = np.where(x['Net_Income'] < lower_bound, lower_bound, x['Net_Income'])

sns.boxplot(x['Net_Income'])

plt.title("Identify Outliers in Net_Income")

plt.show()


# In[55]:


sns.boxplot(x['Cash_Flow'])


# In[56]:


Q1 = x['Cash_Flow'].quantile(0.25)

Q3 = x['Cash_Flow'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Cash_Flow'] = np.where(x['Cash_Flow'] > upper_bound, upper_bound, x['Cash_Flow'])

x['Cash_Flow'] = np.where(x['Cash_Flow'] < lower_bound, lower_bound, x['Cash_Flow'])

sns.boxplot(x['Cash_Flow'])

plt.title("Identify Outliers in Cash_Flow")

plt.show()


# In[57]:


sns.boxplot(x['Assets'])


# In[58]:


Q1 = x['Assets'].quantile(0.25)

Q3 = x['Assets'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Assets'] = np.where(x['Assets'] > upper_bound, upper_bound, x['Assets'])

x['Assets'] = np.where(x['Assets'] < lower_bound, lower_bound, x['Assets'])

sns.boxplot(x['Assets'])

plt.title("Identify Outliers in Assets")

plt.show()


# In[60]:


sns.boxplot(x['Liabilities'])


# In[61]:


Q1 = x['Liabilities'].quantile(0.25)

Q3 = x['Liabilities'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Liabilities'] = np.where(x['Liabilities'] > upper_bound, upper_bound, x['Liabilities'])

x['Liabilities'] = np.where(x['Liabilities'] < lower_bound, lower_bound, x['Liabilities'])

sns.boxplot(x['Liabilities'])

plt.title("Identify Outliers in Liabilities")

plt.show()


# In[62]:


sns.boxplot(x['Equity'])


# In[63]:


Q1 = x['Equity'].quantile(0.25)

Q3 = x['Equity'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Equity'] = np.where(x['Equity'] > upper_bound, upper_bound, x['Equity'])

x['Equity'] = np.where(x['Equity'] < lower_bound, lower_bound, x['Equity'])

sns.boxplot(x['Equity'])

plt.title("Identify Outliers in Equity")

plt.show()


# In[64]:


sns.boxplot(x['Current_Ratio'])


# In[65]:


Q1 = x['Current_Ratio'].quantile(0.25)

Q3 = x['Current_Ratio'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Current_Ratio'] = np.where(x['Current_Ratio'] > upper_bound, upper_bound, x['Current_Ratio'])

x['Current_Ratio'] = np.where(x['Current_Ratio'] < lower_bound, lower_bound, x['Current_Ratio'])

sns.boxplot(x['Current_Ratio'])

plt.title("Identify Outliers in Current_Ratio")

plt.show()


# In[66]:


sns.boxplot(x['Shares_Outstanding'])


# In[67]:


Q1 = x['Shares_Outstanding'].quantile(0.25)

Q3 = x['Shares_Outstanding'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

x['Shares_Outstanding'] = np.where(x['Shares_Outstanding'] > upper_bound, upper_bound, x['Shares_Outstanding'])

x['Shares_Outstanding'] = np.where(x['Shares_Outstanding'] < lower_bound, lower_bound, x['Shares_Outstanding'])

sns.boxplot(x['Shares_Outstanding'])

plt.title("Identify Outliers in Shares_Outstanding")

plt.show()


# In[69]:


plt.figure(figsize=(25,20))

sns.boxplot(x)

plt.show()


# # Examine data insights visually

# In[85]:


# Correlation matrix

corr_matrix = x.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[94]:


# Distribution of each feature

for col in df.columns:
    plt.figure(figsize=(10, 3))
    sns.distplot(df[col], kde=False)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[78]:


# Time series plots


plt.figure(figsize=(12, 6))
plt.plot(x['Date'], df['Open'], label='Open')
plt.plot(x['Date'], df['High'], label='High')
plt.plot(x['Date'], df['Close'], label='Close')
plt.legend()
plt.show()


# In[80]:


# Scatter plots
plt.figure(figsize=(12, 6))
plt.scatter(x['Revenue'], x['Gross_Profit'])
plt.xlabel('Revenue')
plt.ylabel('Gross Profit')
plt.show()


# In[82]:


plt.figure(figsize=(12, 6))
plt.scatter(x['Market_Cap'], x['Net_Income'])
plt.xlabel('Market Cap')
plt.ylabel('Net Income')
plt.show()


# In[96]:


# Box plots

sns.boxplot(x['Current_Ratio'])
sns.boxplot(x['Shares_Outstanding'])


# In[84]:


# Pairplot

sns.pairplot(x[['Open', 'High', 'Close', 'Volume']])


# These plots and statistics will help derive insights from the data, such as:
# 
# - Trends and patterns in stock prices and financial metrics
# - Relationships between different variables
# - Distribution of financial metrics
# - Outliers and anomalies in the data
# 

# In[ ]:




