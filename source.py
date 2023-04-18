#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# Salary estimation in tech jobs with experience
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# Estimate relation between salary with age and experience
# 

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# Do people with higher experience earn more in tech jobs?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# With more years of experience there is higher chance of having more salary

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# kaggle 
# stack overflow survey data 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[156]:


# Start your code here
import pandas as pd
import numpy as np
import opendatasets as od
import requests
import plotly.express as px
from bs4 import BeautifulSoup
import seaborn as sns
from pandas.plotting import scatter_matrix


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Importing Datasetno.1

# In[146]:


df = pd.read_csv(r'C:\stack-overflow-developer-survey-2022\survey_results_public.csv')
df.head()


# Filter the table to just get the required data

# In[147]:


cols_to_keep = ['Employment', 'EdLevel', 'DevType', 'Country','YearsCode','ConvertedCompYearly']
new_df = df.loc[:, cols_to_keep]
new_df.head()


# Look for null values

# In[148]:


new_df.isna().sum()


# Data cleaning: Drop all the null values

# In[149]:


new_df.dropna(inplace=True)
new_df.head()


# Checked if we have any null values remaining

# In[150]:


new_df.isnull().sum()


# Getting data info

# In[142]:


new_df.info()


# Data description

# In[151]:


new_df.describe()


# In[175]:


tech_jobs_df = pd.read_csv(r'C:\Users\diksh\Downloads\salaries_clean.csv')
df.head()


# In[176]:


tech_jobs_df.isnull().sum()


# Importing second dataset

# In[133]:


df = pd.read_csv(r'C:\Users\diksh\Downloads\Salary_Data.csv')
df.head()


# Histogram plt for the dataset

# In[60]:


salaries_data_df.hist(bins=50, figsize=(20,15))
plt.show()


# Scatter plot for a dataset

# In[65]:


salaries_data_df.plot.scatter(x='YearsExperience',y='Salary')


# Finding the correlation Matrix

# In[71]:


corr_matrix = salaries_data_df.corr()
print(corr_matrix)


# Heatmap of the dataset

# In[72]:


sns.heatmap(corr_matrix)


# In[77]:


pd.plotting.scatter_matrix(salaries_data_df)
plt.suptitle('Scatter Matrix of Variables')
plt.show()


# In[82]:


X = salaries_data_df[['YearsExperience']]  
y = salaries_data_df['Salary']  

model = LinearRegression().fit(X, y)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


# In[84]:


X = salaries_data_df[['YearsExperience']]  
y = salaries_data_df['Salary']  

model = LinearRegression().fit(X, y)

plt.scatter(salaries_data_df[['YearsExperience']], salaries_data_df['Salary'])
plt.xlabel('Education')
plt.ylabel('Income')
plt.title('Scatter Plot of Education vs Income')

plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.show()


# In[88]:


X = salaries_data_df[['YearsExperience']]  
y = salaries_data_df['Salary']   

model = LinearRegression().fit(X, y)

new_experience = 20
predicted_salary = model.predict([[new_experience]])

print(f'Predicted income for {new_experience} years of education: ${predicted_salary[0]:,.8f}')


# Regression
# Train the model using sklearn

# In[187]:


X = np.array(salaries_data_df['Age']).reshape(-1, 1)
y = np.array(salaries_data_df['Salary']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)


# In[188]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[189]:


print("Coefficient Of the line", model.coef_)
print("Intercept Of the model", model.intercept_)

score = model.score(X_test, y_test)
print("Score of the model", score)
print("Accuracy", str(score * 100) + '%')


# In[190]:


y_predict = model.predict(X_test)


# In[191]:


plt.scatter(X_train, y_train, color='r')
plt.plot(X_train, model.predict(X_train), color='b')
plt.title('Salary vs Age (Training Set)')
plt.xlabel('Salary')
plt.ylabel('Exp')
plt.show()

plt.scatter(X_test, y_test, color='r')
plt.plot(X_test, model.predict(X_test), color='b')
plt.title('Salary vs Age (Test Set)')
plt.xlabel('Salary')
plt.ylabel('Exp')
plt.show()


# In[192]:


plt.figure(figsize = (12, 6))
plt.scatter(y_test, y_predict, color='r', linestyle='-')
plt.xlabel('y from test set')
plt.ylabel('y from predict set')
plt.show()


# In[193]:


c = [i for i in range(1, len(y_test)+1, 1)]
plt.plot(c, y_test, color='r', linestyle='-')
plt.plot(c, y_predict, color='b', linestyle='-')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Prediction')

plt.show()


# In[194]:


c = [i for i in range(1, len(y_test)+1, 1)]
y = y_test-y_predict
plt.plot(c, y, color='green', linestyle='-')
plt.xlabel('Age')
plt.ylabel('Error')
plt.title('Error Between Preduction and Test Data Set')

plt.show()


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# kaggle
# google

# In[200]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

