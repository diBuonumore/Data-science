#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

raw_data=pd.read_csv("Weather Data.csv")
weather_data = raw_data.copy()

#CATEGORICAL VARIABLES

#typo
weather_data.rename(columns={'Longitute':"Longitude"},inplace=True)

#to make sure all states/provinces has space at the beginning
for index, state in enumerate(weather_data['State/Province']):
    if state == 'Australian Capital Territory':
        weather_data.loc[index, 'State/Province'] = ' Australian Capital Territory'

#remove ID column
weather_data = weather_data.drop(['row ID'],axis=1)

#remove rows with null values from WindGustDir
weather_data = weather_data.dropna(subset=['WindGustDir'])

#fill rows with null values with the mode for WindDir9am and WindDir3pm
mode_winddir9am = weather_data['WindDir9am'].mode().iloc[0]
mode_winddir3pm = weather_data['WindDir3pm'].mode().iloc[0]
weather_data['WindDir9am'] = weather_data['WindDir9am'].fillna(mode_winddir9am)
weather_data['WindDir3pm'] = weather_data['WindDir3pm'].fillna(mode_winddir3pm)

#Cloud9am
missing_indices = weather_data['Cloud9am'].isnull()
num_missing = missing_indices.sum()
non_missing_data = weather_data.loc[~missing_indices,'Cloud9am']
probabilities = non_missing_data.value_counts(normalize=True) # the relative frequencies (probabilities) instead of the actual counts.
random_values = np.random.choice(probabilities.index, size=num_missing, p=probabilities.values)
weather_data.loc[missing_indices,'Cloud9am'] = random_values

#Cloud3pm
missing_indices = weather_data['Cloud3pm'].isnull()
num_missing = missing_indices.sum()
non_missing_data = weather_data.loc[~missing_indices,'Cloud3pm']
probabilities = non_missing_data.value_counts(normalize=True) # the relative frequencies (probabilities) instead of the actual counts.
random_values = np.random.choice(probabilities.index, size=num_missing, p=probabilities.values)
weather_data.loc[missing_indices,'Cloud3pm'] = random_values

#RainToday
weather_data = weather_data[weather_data['RainToday'].notnull()]

weather_data_partly_preprocessed = weather_data.copy()

#NUMIERICAL VARIABLES

#Drop almost all remaining rows with NaNs (without Evaporation and Sunshine)
weather_data_partly_preprocessed = weather_data_partly_preprocessed.dropna(subset=['Pressure9am','Humidity3pm','Humidity9am','WindSpeed9am','MinTemp','MaxTemp','WindSpeed3pm','Pressure3pm'])

def random_imputation(df,variable):
    
    observed_data = df[variable].dropna()
    missing_indices = df[variable].isnull()

    # Fit a kernel density estimate to the observed data
    kde = gaussian_kde(observed_data)

    # Get the number of missing values
    num_missing = missing_indices.sum()

    # Generate random values from the KDE
    random_values = np.round(np.abs(kde.resample(num_missing)[0]),1) # at the beginning I did not ensure that my outputs remain positive, np.abs was important step! (now the distributions are similar, before not really)

    # Fill missing values with random values
    df.loc[missing_indices, variable] = random_values
    
    return df


#Sunshine 
weather_data_partly_preprocessed = random_imputation(weather_data_partly_preprocessed, 'Sunshine')

#Evaporation 
weather_data_partly_preprocessed = random_imputation(weather_data_partly_preprocessed, 'Evaporation')

weather_data_preprocessed = weather_data_partly_preprocessed.copy()

# weather_data_preprocessed.isnull().sum()
#weather_data_preprocessed.info()

weather_data_preprocessed.to_csv('Preprocessing completed-Australian Rainfall.csv',index=False)

