#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno


# In[2]:


raw_data=pd.read_csv("Weather Data.csv")
raw_data.head()


# In[3]:


weather_data = raw_data.copy()
weather_data = weather_data.drop(['row ID'],axis=1)
weather_data.rename(columns={'Longitute':"Longitude"},inplace=True)
for index, state in enumerate(weather_data['State/Province']):
    if state == 'Australian Capital Territory':
        weather_data.loc[index, 'State/Province'] = ' Australian Capital Territory'
weather_data.info()
# 25 columns, 99516 rows
#there are some nulls 
#types: object, float64,int64


# In[4]:


weather_data.shape


# In[5]:


pd.set_option('display.max_columns', None) #to see all the columns; None - meaning there will be no limit to the number of columns displayed
weather_data.describe(include="all")


# In[6]:


#Missing values
NA_data = weather_data.isnull().sum()
NA_data


# In[7]:


NA_percentage = round(NA_data / weather_data.shape[0] *100,1)


# In[8]:


print("Percentage of missing values in each column: \n{}".format(NA_percentage))


# In[ ]:


#Insights: 
#1. Are the missing values clustered in specific regions? 
#2. Better to start from numeric or categorical features? 


# In[9]:


#The code below will display a heatmap where missing values are represented as white vertical bars in the corresponding columns. The more missing values there are in a column, the wider the white bar will be.
#he missingno matrix plot consists of two parts:
#The main part: This is the heatmap that displays the missing data pattern for each variable (column). 
#The rightmost column: This column represents the completeness of each row in the dataset. Each row in the dataframe corresponds to an observation, and the number in this column indicates the count of non-missing values in that row.
msno.matrix(weather_data)
plt.show()


# In[10]:


msno.bar(weather_data)
plt.show()


# In[11]:


#missing data correlation
#if there are high correlations between missing values in different variables, it suggests a pattern in the missing data.
msno.heatmap(weather_data)


# **MCAR - Missing Completely At Random** -  **missingless of data is unrelated to any observed/unobserved variables**:
# 
#     1. Data Entry Errors: In a large dataset, missing values may occur randomly due to data entry errors, such as typos or accidental omissions. For example, a survey respondent might forget to answer a question or an operator may fail to record a data point.
# 
#     2. Technical Issues: Missing data can occur randomly due to technical issues during data collection or recording. For instance, a sensor malfunction in an IoT device may lead to missing data points.
# 
#     3. Survey Non-Response: In surveys, some respondents may choose not to answer certain questions, either intentionally or unintentionally. If the decision to skip a question is unrelated to the respondent's characteristics or the question itself, it can be considered MCAR
# 
# **MAR - Missing At Random - occurs when the probability of **missing data is related to the observed variables** in the dataset but not to the unobserved (missing) data itself:**
# 
#     1. Education Level and Salary: Suppose there is a survey conducted to collect data on individuals' education level and salary. However, individuals with higher education levels are more likely to withhold their salary information due to privacy concerns or other reasons. In this case, the missingness of salary data is related to the observed variable "education level" (higher education level is associated with missing salary data). The missingness is not entirely random but depends on the observed variable (education level).   
#         
# **MNAR - Missing Not At Random** - occurs when the probability of **missing data is related to the unobserved (missing) data itself**, even after considering the observed variables:
# 
#     1.Income Data: In a survey about income, high-income individuals might be less likely to disclose their exact income, leading to missing data for high-income respondents.
#         
#     2.Health Surveys: In a health survey, participants might be more likely to skip questions about sensitive health issues, such as mental health or substance abuse, leading to missing data for those questions. 
# 
#     3.Job Satisfaction: In a survey about job satisfaction, employees who are extremely dissatisfied with their jobs might be less likely to respond truthfully about their level of satisfaction.
# 
#     4.Drug Usage Data: In a study on drug usage, individuals who use illegal drugs might be less likely to report their drug usage accurately due to for example fear of legal consequences

# ## Dealing with missing values
# Common strategies: 
# 
#     1. remove rows/columns with NaNs - however this can lead to a loss of valuable information
#     2. use statistical methods - for time-series data -> interpolation,extrapolation
#     3. instead of filling in missing values, we can create an additional binary column to indicate whether a value is missing or not
#     4. Imputation - replace missing values with estimated/predicted values
# 
# There are several methods to fill gaps in a categorical variable:
# 
# 1. Mode Imputation: Replace the missing values with the most frequent category (mode) in the variable. 
# 
# 2. Random Imputation: Replace the missing values with random samples from the distribution of the existing categories in the variable. 
# 
# 3. Backward Fill (bfill) and Forward Fill (ffill): Fill the missing values with the previous (bfill) or next (ffill) non-missing value in the variable - when the data follows a temporal or sequential pattern.
# 
# 4. K-Nearest Neighbors (KNN) Imputation: based on the similarity of other data points. 
# 
# 5. Hot Deck Imputation: Replace missing values with values randomly selected from similar records based on certain criteria, such as matching values of other features.
# 
# 6. Imputation Using Machine Learning Models like decision trees or random forests to predict the missing categories based on other features.
# 

# # Categorical variables

# ## <span style='color: #3F2E3E;'>Location - nominal</span>
# #### <span style='color: #A78295;'>Name of the city from Australia </span>

# In[12]:


weather_data['Location'].value_counts()
#Why are there the most observations from Canberra,Sydney,Perth,Hobart..? Why this kind of hirarchy? How did they get this proportions?
#Well.. Firstly, these cities are the biggest, the most populous. Secondly, I noticed that the first listed cities are located in different regions on the map. 
#We distinguish 9 states/provinces, and the first 7 cities belongs to the 7 different regions. Circumstance? 


# In[13]:


#type(weather_data['Location'].value_counts()) #pandas.core.series.Series
#weather_data['Location'].value_counts().index
#weather_data['Location'].value_counts().index[:10]
#weather_data['Location'].value_counts()[:10]

location_counts = weather_data['Location'].value_counts()
first_names = location_counts.index[:10]
first_counts = location_counts[:10]

sns.barplot(x=first_names, y=first_counts,palette='viridis')

plt.xlabel('Location')
plt.ylabel('Count')
plt.title('Top 10 Locations by count')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

sns.despine()  # Remove the top and right spines

# Add annotations for each bar
for index, value in enumerate(first_counts):
    plt.text(index, value, str(value), ha='center', va='bottom')
    
# e.g.if first_counts is [10, 25, 15, 30, 20], the enumerate() function will yield the following tuples: (0, 10), (1, 25), (2, 15), (3, 30), and (4, 20).
# index: The x-coordinate at which the text will be placed
# value: The y-coordinate at which the text will be placed
# str(value): The text that will be displayed at the specified coordinates
# ha - horizontal alignment
# va - vertical alignment
plt.show()


# ## <span style='color: #3F2E3E;'>State/Province - nominal</span>
# #### <span style='color: #A78295;'>State or province of the cities in Australia </span>

# In[14]:


weather_data['State/Province'].value_counts()


# In[15]:


state_counts = weather_data['State/Province'].value_counts()

plt.figure(figsize=(8, 6))  #specifies the width and height of the figure in inches
sns.set_palette('tab20b')  

plt.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', startangle=140)

# autopct='%1.1f%%' - automatic percentage -> to display the percentage of each wedge(category)
# f - float, 1.1 - one digit after decimal point, %% - escape sequence to display % symbol
# By default, the first wedge starts from the positive x-axis (0 degrees) and proceeds counterclockwise - startangle

plt.title('State/Province Distribution')

plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.show()


# ## <span style='color: #3F2E3E;'>WindGustDir - nominal</span>
# #### <span style='color: #A78295;'>The direction of the strongest gust during a particular day. (16 compass points)</span>

# In[16]:


pd.set_option('display.max_rows', None)
# weather_data[weather_data['WindGustDir'].isnull()].head(10)
#In rows with missing WindGustDirection there are also NaN values in other variables
#NaN values in WindGustDir represent 6.6% of total number of rows, this is not so much, so that I decided to remove rows 
#with missing values in WindGustDir


# In[17]:


weather_data_cleaned_gust_dir = weather_data.copy()
weather_data_cleaned_gust_dir = weather_data_cleaned_gust_dir.dropna(subset=['WindGustDir'])
NA_data_cleaned_gust_dir = weather_data_cleaned_gust_dir.isnull().sum()
NA_data_cleaned_gust_dir


# In[18]:


#we removed the following number of missing observations:
NA_data - NA_data_cleaned_gust_dir


# In[19]:


weather_data_cleaned_gust_dir['WindGustDir'].value_counts()


# In[20]:


custom_order = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
reversed_custom_order = custom_order[::-1]
wind_gust_dir = weather_data_cleaned_gust_dir['WindGustDir'].value_counts().reindex(reversed_custom_order)
wind_gust_dir


# In[21]:


plt.figure(figsize=(8, 6))  #specifies the width and height of the figure in inches
sns.set_palette('tab20b')  

plt.pie(wind_gust_dir, labels=wind_gust_dir.index, autopct='%1.1f%%', startangle=100)

plt.title('The direction of the strongest gust during a particular day - frequency',pad=20) # pad=20 to make space between the title and the pie chart

plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.show()


# ## <span style='color: #3F2E3E;'>WindDir9am / WindDir3pm - nominal</span>
# #### <span style='color: #A78295;'>The direction of the wind for 10 min prior to 9 am. / 3pm. (compass points)</span>

# In[22]:


# After removing NaN rows in WindGustDir, there is NaNs rows in: 
# WindDir9am         5265
# WindDir3pm          668
weather_data_cleaned_gust_dir.dropna(subset=['WindDir9am']).isnull().sum()
#I want to check if I remove missing values from WindDir9am, the missing values from WindDir3pm will disappear too. But only around 200 records contained NANs in both variables.  


# In[23]:


NA_percentage = round(NA_data_cleaned_gust_dir / weather_data_cleaned_gust_dir.shape[0] *100,1)
print("Percentage of missing values in each column: \n{}".format(NA_percentage))


# In[24]:


weather_data_dir = weather_data_cleaned_gust_dir.copy()


# In[25]:


# Before filling null values
wind_dir_9am = weather_data_dir['WindDir9am'].value_counts().reindex(custom_order)
wind_dir_3pm = weather_data_dir['WindDir3pm'].value_counts().reindex(custom_order)
wind_dir = pd.DataFrame({'Direction': custom_order,'10 min prior to 9 am': wind_dir_9am.values, '10 min prior to 3 pm':wind_dir_3pm.values})
wind_dir.set_index('Direction', inplace=True)
ax = wind_dir.plot(kind='bar', figsize=(10, 6), width=0.5,color=['#3F4E4F', '#DCD7C9'], edgecolor='black')
ax.set_ylabel('Values')
ax.set_xlabel('Wind Direction')
ax.set_title('Wind Direction Comparison (9am vs. 3pm)')
plt.legend(title='Time', fontsize=10)
plt.xticks(rotation=45)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.show()


# In[ ]:


# After filling null values


# In[26]:


#A convenient solution in this case may be to replace NaNs with the most frequent category (mode) since we have only few missing values
#type(weather_data_cleaned['WindDir9am'].mode()) pandas.core.series.Series
weather_data_dir_with_mode = weather_data_dir.copy()
mode_winddir9am = weather_data_dir_with_mode['WindDir9am'].mode().iloc[0]
mode_winddir3pm = weather_data_dir_with_mode['WindDir3pm'].mode().iloc[0]
print(f"The mode in WindDri9am is: {mode_winddir9am}, while in WindDir3pm: {mode_winddir3pm}")


# In[27]:


weather_data_dir_with_mode['WindDir9am'] = weather_data_dir_with_mode['WindDir9am'].fillna(mode_winddir9am)
weather_data_dir_with_mode['WindDir3pm'] = weather_data_dir_with_mode['WindDir3pm'].fillna(mode_winddir3pm)


# In[28]:


weather_data_dir_with_mode.isnull().sum()


# In[29]:


wind_dir_9am = weather_data_dir_with_mode['WindDir9am'].value_counts().reindex(custom_order)
wind_dir_3pm = weather_data_dir_with_mode['WindDir3pm'].value_counts().reindex(custom_order)


# In[ ]:


#doubled_custom_order = [dir for dir in custom_order for _ in range(2)]


# In[30]:


wind_dir = pd.DataFrame({'Direction': custom_order,'10 min prior to 9 am': wind_dir_9am.values, '10 min prior to 3 pm':wind_dir_3pm.values})
wind_dir


# In[31]:


wind_dir.set_index('Direction', inplace=True)
wind_dir


# In[32]:


# Plot the grouped barplot
ax = wind_dir.plot(kind='bar', figsize=(10, 6), width=0.5,color=['#3F4E4F', '#DCD7C9'], edgecolor='black')
ax.set_ylabel('Values')
ax.set_xlabel('Wind Direction')
ax.set_title('Wind Direction Comparison (9am vs. 3pm)')
plt.legend(title='Time', fontsize=10)
plt.xticks(rotation=45)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.show()


# ## <span style='color: #3F2E3E;'>Cloud9am / Cloud3pm - ordinal</span>
# #### <span style='color: #A78295;'>Cloud-obscured portions of the sky at 9 am. / 3pm.(scale in oktas - eighths)</span>

# In meteorology, cloud cover is often measured in "oktas." An okta is a unit of measurement used to estimate the fraction of the sky covered by clouds at any given time. It is divided into 8 equal parts, with each okta representing 1/8th of the sky covered by clouds.However, we can distinguish also 0 oktas and 9 oktas. 
# 
# Here's how cloud cover is typically described in terms of oktas:
# 
# 0 oktas: Completely clear sky, no clouds.
# 
# 1 okta: Very few clouds, almost clear sky.
# 
# 2 oktas: Partly cloudy, 25% of the sky covered by clouds.
# 
# 3 oktas: Mostly cloudy.
# 
# 4 oktas: More clouds, about half of the sky covered.
# 
# 5 oktas: Overcast sky.
# 
# 6 oktas: Cloudy, with about 75% of the sky covered by clouds.
# 
# 7 oktas: Mostly covered.
# 
# 8 oktas: Completely overcast, the entire sky covered by clouds.
# 
# 9 oktas: Sky obscured by thick clouds.
# 

# In[33]:


weather_data_clouds = weather_data_dir_with_mode.copy()
#weather_data_clouds.info()


# In[34]:


weather_data_clouds[['Cloud9am','Cloud3pm']].head(10)


# In[35]:


sorted(weather_data_clouds['Cloud3pm'].unique())


# In[36]:


sorted(weather_data_clouds['Cloud9am'].unique())


# In[37]:


weather_data_clouds[['Cloud9am','Cloud3pm']].isnull().sum() / weather_data_clouds.shape[0] * 100 


# In[38]:


weather_cloud_9am = weather_data_clouds['Cloud9am'].value_counts() 
weather_cloud_3pm = weather_data_clouds['Cloud3pm'].value_counts()
weather_cloud_9am.index = weather_cloud_9am.index.astype(int)
weather_cloud_3pm.index = weather_cloud_3pm.index.astype(int)
clouds = pd.DataFrame({'Clouds at 9am': weather_cloud_9am, 'Clouds at 3pm':weather_cloud_3pm})
ax = clouds.plot(kind='bar', figsize=(10, 6), width=0.5,color=['#DBDFEA', '#ACB1D6'], edgecolor='black')
ax.set_ylabel('Values')
ax.set_xlabel('Cloud cover')
ax.set_title('Cloudiness measurement at 9am and 3pm')
plt.legend(title='Time', fontsize=10)
plt.xticks(rotation=0)
sns.despine()
plt.show()


# ## Random Imputation

# In[ ]:


# Cloud9am


# In[39]:


#Before
plt.figure(figsize=(8, 6))
sns.countplot(x='Cloud9am', data=weather_data_clouds)
plt.xlabel('Cloud9am')
plt.ylabel('Count')
plt.title('Cloud9am Value Counts')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# def fill_categorical_gaps(data,col):
#     if data[col].dtype == 'object':  # Check if the column is categorical
#         missing_indices = data[col].isnull()
#         non_missing_data = data.loc[~missing_indices, col]
#         probabilities = non_missing_data.value_counts(normalize=True)
#         num_missing = missing_indices.sum()
        
#         # Generate random values based on the probabilities
        
#         random_values = np.random.choice(probabilities.index, size=num_missing, p=probabilities.values)
#         data.loc[missing_indices, col] = random_values

#     return data

# fill_categorical_gaps(weather_data_clouds,'Cloud9am')


# In[40]:


missing_indices = weather_data_clouds['Cloud9am'].isnull()
num_missing = missing_indices.sum()
num_missing


# In[41]:


non_missing_data = weather_data_clouds.loc[~missing_indices,'Cloud9am' ]
non_missing_data.head(5)


# In[42]:


probabilities = non_missing_data.value_counts(normalize=True) # the relative frequencies (probabilities) instead of the actual counts.
random_values = np.random.choice(probabilities.index, size=num_missing, p=probabilities.values)
weather_data_clouds.loc[missing_indices,'Cloud9am'] = random_values
weather_data_clouds['Cloud9am'].value_counts()


# In[43]:


#After
plt.figure(figsize=(8, 6))
sns.countplot(x='Cloud9am', data=weather_data_clouds)
plt.xlabel('Cloud9am')
plt.ylabel('Count')
plt.title('Cloud9am Value Counts')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Cloud 3pm


# In[44]:


#Before
plt.figure(figsize=(8, 6))
sns.countplot(x='Cloud3pm', data=weather_data_clouds)
plt.xlabel('Cloud9am')
plt.ylabel('Count')
plt.title('Cloud9am Value Counts')
plt.xticks(rotation=45)
plt.show()


# In[45]:


missing_indices = weather_data_clouds['Cloud3pm'].isnull()
num_missing = missing_indices.sum()
non_missing_data = weather_data_clouds.loc[~missing_indices,'Cloud3pm']

probabilities = non_missing_data.value_counts(normalize=True) # the relative frequencies (probabilities) instead of the actual counts.
random_values = np.random.choice(probabilities.index, size=num_missing, p=probabilities.values)

weather_data_clouds.loc[missing_indices,'Cloud3pm'] = random_values


# In[46]:


#After
plt.figure(figsize=(8, 6))
sns.countplot(x='Cloud3pm', data=weather_data_clouds)
plt.xlabel('Cloud3pm')
plt.ylabel('Count')
plt.title('Cloud3pm Value Counts')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#weather_data_clouds.isnull().sum()


# #### I have tried also with other methods,namely with the mode and RandomForestClassifier but I was not satisfied with the results. 

# #### Ultimately, I chose filling NaNs with random values based on the distribution of the variable.

# ## Filling gaps with the mode

# In[ ]:


# weather_data_clouds = weather_data_dir_with_mode.copy()
# clouds_9am_mode = weather_data_clouds['Cloud9am'].mode().iloc[0]
# clouds_3pm_mode = weather_data_clouds['Cloud3pm'].mode().iloc[0]

# weather_data_clouds['Cloud9am']= weather_data_clouds['Cloud9am'].fillna(clouds_9am_mode)
# weather_data_clouds['Cloud9am']= weather_data_clouds['Cloud9am'].fillna(clouds_9am_mode)

# weather_cloud_9am_with_mode = weather_data_clouds['Cloud9am'].value_counts()
# weather_cloud_3pm_with_mode = weather_data_clouds['Cloud3pm'].value_counts()

# clouds_with_mode = pd.DataFrame({'Clouds at 9am': weather_cloud_9am_with_mode, 'Clouds at 3pm':weather_cloud_3pm_with_mode})
# clouds_with_mode

# ax = clouds_with_mode.plot(kind='bar', figsize=(10, 6), width=0.5,color=['#DBDFEA', '#ACB1D6'], edgecolor='black')
# ax.set_ylabel('Values')
# ax.set_xlabel('Cloud cover')
# ax.set_title('Cloudiness measurement at 9am and 3pm \n after filling NaNs with mode')
# plt.legend(title='Time', fontsize=10)
# plt.xticks(rotation=0)
# sns.despine()
# plt.show()


# # Random Forest Classifier
# Random Forests are relatively resistant to null values and can handle missing data implicitly during the tree-building process.
# RandomForestClassifier is a supervised learning algorithm - the model is trained on labeled data -> 
# the input data (features) and their corresponding output labels (target) are provided during the training process.
# 
# Inputs (features) have to be converted to numeric - for categorical features I will use the target encoding (one-hot enocoding will cause too many classes)
# Will it be good idea to combine target encoding with one-hot encoding? 
# 
# Outputs should remain categorical 
# 
# Target Leakage: "Ensure that the target encoding is calculated based only on the training set and not using any information from the validation or test sets."

# Actually, from the very beginning I wanted to use RandomForestClassifier algorithm (method that consist of multiple Decision Trees), however it gave me less than 50% accuracy on test data and validation data. I tried to improve it by experimenting with: 
# 1) hyperparametrs like number of decision trees (n_estimators), maximum depth of trees(max_depth),minimum number of samples required to split a node (min_samples_split).
# 2) cross-validation 
# 3) class weights - it is recommended to set the class_weight parameter to "balanced" in RFC if there is class imbalance in the target variable
# 4) I was thinking about using one of the hyperparameter optimization techniques (e.g., grid search, random search, Bayesian optimization) to automatically search for the optimal number of trees, but for some reasons I could not run it. 
# 5) smoothing -> to enhence the target encoding process: to handle rare categories
# 
# I think **overfitting** is a major problem here but somehow I can't figure out the reason of it. I tried to optimalize both target encoding and RFC model and there goes something wrong. For now, I have no idea what. The code that I've tried is presented below.

# In[47]:


from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data_encoded = weather_data_dir_with_mode.copy()

# Conversion into categorical types (because in fact RainTomorrow, Cloud9am/3pm are categorical)
data_encoded['RainTomorrow'] = data_encoded['RainTomorrow'].map({0: 'No', 1: 'Yes'})
cloud_categories = {
    0: 'Clear',
    1: 'Few Clouds',
    2: 'Partly Cloudy',
    3: 'Mostly Cloudy',
    4: 'Cloudy',
    5: 'Overcast',
    6: 'Obscured',
    7: 'Mostly Obscured',
    8: 'Completely Overcast',
    9: 'Sky Obscured'
}
data_encoded['Cloud3pm'] = data_encoded['Cloud3pm'].map(cloud_categories).astype('category')

# Remove all rows with NaNs + remember to reset the indexes, later their might be problematic !!
data_encoded = data_encoded.dropna(how='any')
data_encoded.reset_index(drop=True, inplace=True)

target_column = 'Cloud9am'
categorical_columns_to_encode = ['Location', 'State/Province', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow', 'Cloud3pm']

X = data_encoded.drop(['Cloud9am'], axis=1)
y = data_encoded['Cloud9am']

# Scale features/inputs - without dummies !!

X_unscaled = X.drop(columns=categorical_columns_to_encode)
scaler = StandardScaler()
scaler.fit(X_unscaled)
X_scaled = scaler.transform(X_unscaled)
X_scaled = pd.DataFrame(X_scaled,columns=X_unscaled.columns)
X = pd.concat([X_scaled, X[categorical_columns_to_encode]], axis=1)
#X = X.drop(["MaxTemp","MinTemp","Temp9am","Pressure3pm"],axis=1)

# Split the data into train,validation,train -> proportion: 70:10:20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# Target encoding - watch that you fit only once, later you just use transform function on different data

for col in categorical_columns_to_encode:
    te = TargetEncoder(smoothing=0.2)
    te.fit(X_train[col], y_train)
    X_train[col] = te.transform(X_train[col])
    X_test[col] = te.transform(X_test[col])
    X_val[col] = te.transform(X_val[col])
    
# RandomForestClassifier

# rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
class_weights = 'balanced'

rf_classifier = RandomForestClassifier(class_weight = class_weights,max_depth = 40, min_samples_split = 5,random_state = 5)
rf_classifier.fit(X_train, y_train)

# Evaluate the model on the training set
training_accuracy = rf_classifier.score(X_train, y_train)
print("Training Accuracy:", training_accuracy)

validation_accuracy = rf_classifier.score(X_val, y_val)
print("Validation Accuracy:", validation_accuracy)

test_accuracy = rf_classifier.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)


# ## Cross-validation

# In[48]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation on the random forest classifier
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)  # 5-fold cross-validation
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


# ## Learning curve 

# In[49]:


from sklearn.model_selection import learning_curve

# Create learning curve with the random forest classifier
train_sizes, train_scores, test_scores = learning_curve(
    rf_classifier, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)

# Calculate mean and standard deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Accuracy", color="blue")
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.1,
    color="blue",
)
plt.plot(train_sizes, test_mean, label="Validation Accuracy", color="red")
plt.fill_between(
    train_sizes,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.1,
    color="red",
)
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.grid()
plt.show()


# ## Attempt to find the best hyperparameters 

# In[ ]:


# Define a range of values for max_depth and min_samples_split to try
#max_depth_range = [10, 20, 30, None]  # None means no maximum depth, allowing trees to grow freely
max_depth_range = [10, 20, 30, None]
min_samples_split_range = [2, 5, 10, 20]

best_validation_accuracy = 0
best_max_depth = None
best_min_samples_split = None

for max_depth in max_depth_range:
    for min_samples_split in min_samples_split_range:
        rf_classifier = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=5)
        rf_classifier.fit(X_train, y_train)
        validation_accuracy = rf_classifier.score(X_val, y_val)
        print(f"Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Validation Accuracy: {validation_accuracy}")


# ## GridSearchCV

# In[ ]:


# from sklearn.model_selection import GridSearchCV

# # Define the hyperparameter grid to search
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
# }

# # Create the GridSearchCV object
# grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # Perform the grid search on the training set
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters and model from the grid search
# best_rf_classifier = grid_search.best_estimator_

# # Evaluate the best model on the validation set
# validation_accuracy = best_rf_classifier.score(X_val, y_val)
# print("Best Validation Accuracy:", validation_accuracy)


# ## Feature importances

# In[50]:


rf_classifier.feature_importances_.shape


# In[51]:


X.columns


# In[52]:


features = X.columns
importances = rf_classifier.feature_importances_
sorted_indices = np.argsort(importances)[::-1] #sort indexes in descending order

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_indices], y=[features[i] for i in sorted_indices])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.show()


# ## Confusion matrix
# 
# A confusion matrix typically has four main components:
# 
# True Positive (TP): The number of instances that are correctly predicted as positive by the model.
# 
# False Positive (FP): The number of instances that are incorrectly predicted as positive by the model when they are actually negative.
# 
# True Negative (TN): The number of instances that are correctly predicted as negative by the model.
# 
# False Negative (FN): The number of instances that are incorrectly predicted as negative by the model when they are actually positive.

# In[53]:


confusion_matrix = pd.DataFrame({'Actual Positive': ['TP', 'FN'], 'Actual Negative': ['FP', 'TN']}, index=['Predicted Positive', 'Predicted Negative'])

print(confusion_matrix)


# In[54]:


from sklearn.metrics import confusion_matrix
y_pred = rf_classifier.predict(X_test)
confusion_matrix(y_test,y_pred)
# It shows how many values are correct classified (those value are in the diagonal)


# ## Classification_report
# Precision: The ability of the model to correctly identify instances of a given class. It is the ratio of true positive predictions to the total predicted positive instances.
# 
# Recall: The ability of the model to correctly identify all instances of a given class. It is the ratio of true positive predictions to the total actual positive instances.
# 
# F1-score: The harmonic mean of precision and recall, providing a balanced measure between precision and recall.
# 
# Support: The number of samples in each class.
# 
# Accuracy: The overall accuracy of the model, which is the ratio of correct predictions to the total number of samples.
# 
# Macro avg: The average of the metrics (precision, recall, F1-score) for all classes. It gives equal importance to each class.
# 
# Weighted avg: The weighted average of the metrics, taking into account the support (number of samples) for each class.

# In[55]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))


# ## <span style='color: #3F2E3E;'>RainToday - nominal</span>
# #### <span style='color: #A78295;'>If today is rainy then ‘Yes’. If today is not rainy then ‘No’.</span>

# In[56]:


# Checkpoint
weather_data_after_clouds = weather_data_clouds.copy()
weather_data_after_clouds['RainToday'].head(5)
weather_data_after_clouds['RainToday'].value_counts() #does not contain NaNs


# In[57]:


weather_data_after_clouds['RainToday'].isnull().sum()/weather_data_after_clouds.shape[0]


# In[58]:


#I'm going to remove all missing values, because they constitute only ~ 0.009%
weather_data_after_clouds = weather_data_after_clouds[weather_data_after_clouds['RainToday'].notnull()]
# 2nd way 
# weather_data_after_clouds= weather_data_after_clouds.dropna(subset=['RainToday'])
weather_data_after_clouds['RainToday'].isnull().sum()


# In[59]:


weather_data_after_clouds.shape


# In[60]:


sns.set(style="whitegrid", palette="pastel")

sns.countplot(x='RainToday', data=weather_data_after_clouds, hue='RainToday',  order=['Yes', 'No'],dodge=False)

plt.xlabel('Rain Today')
plt.ylabel('Count')
plt.title('Count of Rain Today')

plt.legend(title='Rain', loc='upper right', labels=['Yes', 'No'])

# Adjust figure size for better visualization
plt.figure(figsize=(6, 4))

plt.show()


# In[ ]:


# weather_data_after_clouds['RainToday'] = weather_data_after_clouds['RainToday'].map({'Yes':1,'No':0}) 
# weather_data_after_clouds.head(10)


# In[61]:


weather_data_rain = weather_data_after_clouds.copy()


# <h3 style='text-align: center;'> OUR MAIN PURPOSE </h3>

# ## <span style='color: #3F2E3E;'>RainTomorrow - nominal</span>
# #### <span style='color: #A78295;'>If tomorrow is rainy then 1. If tomorrow is not rainy then 0 </span>

# In[62]:


weather_data_rain['RainTomorrow'].head(5)
weather_data_rain['RainTomorrow'].value_counts()


# In[63]:


weather_data_rain['RainToday'].value_counts() # Very similar to Rain Tomorrow -> Rain Today is a key variable for tomorrow rain prediction


# In[64]:


weather_data_rain['RainTomorrow'].isnull().sum()


# In[65]:


sns.set(style="whitegrid", palette="pastel")

sns.countplot(x='RainTomorrow', data=weather_data_rain, hue='RainTomorrow',dodge=False,order=[1,0])

plt.xlabel('Rain Tomorrow')
plt.ylabel('Count')
plt.title('Distribution of Rain Tomorrow')

plt.legend(title='Rain', loc='upper right', labels=['Yes', 'No'])

# Set custom labels for x-axis, position 0 -> yes, position 1 -> no 
plt.xticks(ticks=[0, 1], labels=['Yes', 'No'])

plt.figure(figsize=(6, 4))

plt.show()

