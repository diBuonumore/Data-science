#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import gaussian_kde


# In[2]:


def load_data():
    raw_data=pd.read_csv("Weather Data.csv")
    weather_data = raw_data.copy()

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
    raw_data = weather_data[weather_data['RainToday'].notnull()]

    #RainTomorrow - completed
    
    return raw_data


# In[3]:


weather_data = load_data()


# In[4]:


weather_data.head()


# In[5]:


weather_data.shape


# In[6]:


round( weather_data.isnull().sum()/ weather_data.shape[0] *100,1)


# In[7]:


#weather_data = weather_data.dropna(subset=['MinTemp','Rainfall','WindSpeed9am','WindSpeed3pm','Humidity9am'])
# Often happens that missing observations occur in the same row in few variables, so that sometimes it is sufficient to 
# remove nulls just from few variables to have completed data
weather_data = weather_data.dropna(subset=['Pressure9am','Humidity3pm','Humidity9am','WindSpeed9am','MinTemp','MaxTemp','WindSpeed3pm','Pressure3pm'])

# weather_data.isnull().sum()/ weather_data.shape[0] *100
weather_data.isnull().sum()
#Attention -> function may show 0% of nulls in the row but in realty there will be some null values, however
#they occupy less than 0.1%  -> so do not use % here or do not round numbers !!! 


# In[8]:


weather_data.shape
# 83899 records still good


# In[9]:


weather_data_partly_preprocessed = weather_data.copy()


# In[10]:


weather_data_partly_preprocessed.info()


# In[11]:


columns_to_exclude = ['Latitude','Longitute','Cloud9am','Cloud3pm','RainTomorrow']
selected_columns = [col for col in weather_data_partly_preprocessed.columns if col not in columns_to_exclude]
weather_data_partly_preprocessed[selected_columns].describe()


# # Numerical variables

# Type numerical: **discrete/continuous** 
# 
# Discrete: can be counted in a finite matter 
# 
# Continuous: impossible to count,infinite (digits after dot)
# 
# Level of measurement: Quantitative: **interval/ratio** 
# 
# Interval: do not have true zero, we may have negative number like degress Celsisu, year (p.n.e) etc.
# 
# Ratio: have true zero, we cannot have negative number like degrees Kelvin,length,distance etc.

# ## Normal distribution? 
# Normality is often an assumption for many statistical tests and models, so that I found it usefull to check whether the data follows normal distribution. 
# 
# How to do it?  vizualization + statistical test
# 
# Vizualization: histogram,Q-Q plots, P-P plots
# 
# Tests:
# 
#     1. for small/medium samples: Shapiro-Wilk Test, D'Agostino-Pearson Test, The Lilliefors test(modification of the Kolmogrov-Smirnow test)
#     2. for larger samples: Kolmogorov-Smirnov Test,Jarque-Bera Test,Anderson-Darling Test
# 
# Additionally, for large samples, even slight deviations from a normal distribution can be detected as statistically significant.   
# Methods that I will be using: 
# 1. histogram (with density curve), boxplot, CDF
# 2. Q-Q plot, P-P plot
# 3. Kolmogorow-Smirnov Test

# In[12]:


def plot_analysis_numerical_variable(dataframe, variable, label_x, title):
    # Set up the figure and subplots with centered alignment
    fig, axes = plt.subplots(nrows=3, figsize=(6, 10), sharex=True, gridspec_kw={'width_ratios': [1], 'hspace': 0.5})

    # Plot the histogram with density curve
    sns.histplot(data=dataframe, x=variable, kde=True, ax=axes[0], color='#4F709C', bins=20)
    axes[0].set(title='Histogram with Density Curve', ylabel='Density')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot the boxplot
    sns.boxplot(data=dataframe, x=variable, ax=axes[1], color='#D8C4B6')
    axes[1].set(title='Boxplot')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot the cumulative distribution
    sorted_data = dataframe[variable].sort_values()  # in ascending order

    # Calculate the cumulative distribution
    cumulative_distribution = (np.arange(len(sorted_data)) + 1) / len(sorted_data)
    axes[2].plot(sorted_data, cumulative_distribution, marker='.', linestyle='none', color='#4F709C', markersize=2)
    # without marker I could write linewidth, because of it I have to specify the width of the curve with markersize

    axes[2].set(title='Cumulative Distribution', xlabel=label_x, ylabel='Cumulative Probability')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].grid(axis='x', linestyle='-', alpha=0.1)

    # Calculate mean and standard deviation
    mean = dataframe[variable].mean()
    std = dataframe[variable].std()

    # Add vertical lines for mean, mean + std, and mean - std
    def std_lines(mean,std,num,color):
        axes[0].axvline(mean + num*std, color= color, linestyle='--', label=f'±{num}σ')
        axes[0].axvline(mean - num*std, color=color, linestyle='--')
    
    axes[0].axvline(mean, color='#BA704F', linestyle='--', label='μ')
    std_lines(mean,std,1,'#5C8984')
    std_lines(mean,std,2,'#CBB279')
    std_lines(mean,std,3,'#2E4F4F')
    std_lines(mean,std,4,'grey')
    axes[0].legend()

    # Manually adjust the subplot parameters for proper alignment
    plt.subplots_adjust(hspace=0.5, top=0.9)  # top parameter adjusts the space for the title

    # Add a common title for all rows (subplots) and align it to the center
    fig.text(0.5, 0.95, title, fontsize=16, ha='center')

    # Show the plot
    plt.show()


# In[13]:


#The KS statistic provides a measure of how well the data fits the theoretical distribution. A smaller KS statistic indicates a better fit between the empirical data and the theoretical distribution. Conversely, a larger KS statistic suggests a poorer fit.
def Kolmogorow_Smirnov_Test(dataframe,variable):
    test_statistic, p_value = stats.kstest(dataframe[variable], 'norm')

    if p_value < 0.05:
        print(f"Test statistic = {test_statistic}, p-value = {p_value} < 0.05: The data is not normally distributed")
    else:
        print(f"Test statistic = {test_statistic}, p-value = {p_value} >= 0.05: The data is normally distributed")


# In[14]:


# probability-probability plot is used to compare CDFs (cumulative distribution functions) of two distributions(empirical and theoretical)
#When you call norm.cdf(sorted_data, loc=mean_empirical, scale=std_dev_empirical), the function will calculate the 
#cumulative distribution function (CDF) for each value in sorted_data based on the normal distribution with the given 
#parameters (mean_empirical and std_dev_empirical).
def P_P_plot(dataframe,variable):
    # Calculate empirical cumulative probability
    sorted_data = dataframe[variable].sort_values()  # in ascending order
    empirical_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Calculate theoretical cumulative probability (for normal distribution)
    mean = np.mean(sorted_data)
    std_dev = np.std(sorted_data)
    theoretical_prob = stats.norm.cdf(sorted_data, loc=mean, scale=std_dev) # both cdfs (theoretical and empirical) have the same mean and std dev

#     to see how theoretical and empirical CDF look  
#     plt.plot(sorted_data, empirical_prob, label='Empirical CDF', linestyle='-', color='blue')
#     plt.plot(sorted_data, theoretical_prob, label='Theoretical CDF', linestyle='-', color='red')

    # Create the plot
    plt.plot(empirical_prob,theoretical_prob,marker='o', linestyle=' ', color='#374259',markersize=1.5)

#     Add a 45-degree reference line
    plt.plot([0, 1], [0, 1], color='#CD1818', linestyle='--', label='45-degree Line')

    # Set plot labels and title
    plt.ylabel('Theoretical Cumulative Probability')
    plt.xlabel('Empirical Cumulative Probability')
    plt.title('P-P plot')
    #plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


# In[15]:


# quantile-quantile (Q-Q) plot to visually compare the data to a theoretical normal distribution.
# we're comparing the quantiles of two distributions
# The x-axis represents the quantiles of the theoretical distribution, the y-axis the quantiles of the observed data.
# If the data follows the theoretical distribution closely, the points on the Q-Q plot will roughly form a straight line.
# Any deviations from the straight line indicate departures from the theoretical distribution.
# Points above the line suggest that the data's tails are heavier (larger values) than the theoretical distribution.
# Points below the line suggest that the data's tails are lighter (smaller values) than the theoretical distribution.
#In a normal distribution, the theoretical quantiles are often denoted as z-scores and represent the number of standard deviations a particular value is from the mean.
def Q_Q_plot(dataframe, variable):
    
    plt.figure(figsize=(8, 6))
    _, axes = plt.subplots()
    _, (__, ___, r) = stats.probplot(dataframe[variable], dist='norm', plot=axes)

    # Customize the marker style and color
    point_color = '#374259'
    line_color ='#CD1818'
    axes.get_lines()[0].set_marker('.')  # Marker for data points
    axes.get_lines()[0].set_color(point_color)  # Color of the data points
    axes.get_lines()[0].set_markerfacecolor(point_color)  # Color of the data point faces
    axes.get_lines()[0].set_markeredgecolor(point_color)  # Color of the data point edges
    axes.get_lines()[1].set_color(line_color)  # Color of the line connecting the points
    axes.get_lines()[1].set_linewidth(2)  # Width of the line connecting the points

    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()


# ## <span style='color: #3F2E3E;'>Temp9am/3pm - continuous - interval</span>
# #### <span style='color: #A78295;'>The temperature at 9 am. / 3pm.(degree Celsius)</span>

# #### Temp9am

# In[16]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Temp9am','Temperature (°C)','Temperature at 9am analysis')


# In[17]:


Kolmogorow_Smirnov_Test(weather_data_partly_preprocessed,'Temp9am')


# In[19]:


P_P_plot(weather_data_partly_preprocessed, 'Temp9am')


# In[20]:


Q_Q_plot(weather_data_partly_preprocessed, 'Temp9am')


# #### Temp3pm
# 

# In[21]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Temp3pm','Temperature (°C)','Temperature at 3pm analysis')


# In[ ]:


Kolmogorow_Smirnov_Test(weather_data_partly_preprocessed,'Temp3pm')


# In[ ]:


P_P_plot(weather_data_partly_preprocessed, 'Temp3pm')


# In[ ]:


Q_Q_plot(weather_data_partly_preprocessed, 'Temp3pm')


# ## <span style='color: #3F2E3E;'>Min/MaxTemp - continuous - interval</span>
# #### <span style='color: #A78295;'>The minimum/maximum temperature during a particular day. (degree Celsius) </span>

# #### MinTemp

# In[22]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'MinTemp','Temperature (°C)','The minimum temperature during a day')


# In[ ]:


Kolmogorow_Smirnov_Test(weather_data_partly_preprocessed,'MinTemp')


# In[ ]:


P_P_plot(weather_data_partly_preprocessed, 'MinTemp')


# In[ ]:


Q_Q_plot(weather_data_partly_preprocessed, 'MinTemp')


# #### MaxTemp

# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'MaxTemp','Temperature (°C)','The maximum temperature during a day')


# In[ ]:


Kolmogorow_Smirnov_Test(weather_data_partly_preprocessed,'MaxTemp')


# In[ ]:


P_P_plot(weather_data_partly_preprocessed, 'MaxTemp')


# In[ ]:


Q_Q_plot(weather_data_partly_preprocessed, 'MaxTemp')


# ## <span style='color: #3F2E3E;'>Rainfall - continuous - ratio </span>
# #### <span style='color: #A78295;'>Rainfall during a particular day. (millimeters)</span>

# In[ ]:


weather_data_partly_preprocessed.describe()


# In[ ]:


weather_data_partly_preprocessed[weather_data_partly_preprocessed['Rainfall'] > 50 ].shape


# In[ ]:


# pd.set_option('display.max_rows', None)
outliers = weather_data_partly_preprocessed[weather_data_partly_preprocessed['Rainfall'] > 50 ]
outliers['State/Province'].value_counts()
# outliers['State/Province'].nunique()


# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Rainfall','Milimeters (mm)','Rainfall during a day')


# In[ ]:


Kolmogorow_Smirnov_Test(weather_data_partly_preprocessed,'Rainfall')


# In[ ]:


P_P_plot(weather_data_partly_preprocessed, 'Rainfall')


# In[ ]:


Q_Q_plot(weather_data_partly_preprocessed, 'Rainfall')


# ## <span style='color: #3F2E3E;'>WindGustSpeed - continuous - ratio</span>
# #### <span style='color: #A78295;'>Speed of strongest gust during a particular day. (kilometers per hour)</span>

# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'WindGustSpeed','Gust speed (km/h)','Speed of strongest gust during a day')


# In[ ]:


# pd.set_option('display.max_rows', None)
# sorted(weather_data_partly_preprocessed['WindGustSpeed'].unique())


# ## <span style='color: #3F2E3E;'>WindSpeed9am/3pm - continuous - ratio</span>
# #### <span style='color: #A78295;'>Speed of the wind for 10 min prior to 9 am. / 3pm. (kilometers per hour)</span>

# #### WindSpeed9am

# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'WindSpeed9am','Wind speed (km/h)','Speed of the wind for 10 min prior to 9 am')


# #### WindSpeed3pm

# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'WindSpeed3pm','Wind speed (km/h)','Speed of the wind for 10 min prior to 3 pm')


# ## <span style='color: #3F2E3E;'>Humidity9am/3pm - continuous - ratio</span>
# #### <span style='color: #A78295;'>The humidity of the wind at 9 am. / 3pm. (percent)</span>

# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Humidity9am','Humidity (%)','The humidity of the wind at 9 am')


# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Humidity3pm','Humidity (%)','The humidity of the wind at 3 pm')


# ## <span style='color: #3F2E3E;'>Pressure9am/3pm - continuous - ratio</span>
# #### <span style='color: #A78295;'> Atmospheric pressure at 9 am. / 3pm. (hectopascals)</span>

# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Pressure9am','Pressure (hPa)','Atmospheric pressure at 9 am')


# In[ ]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Pressure3pm','Pressure (hPa)','Atmospheric pressure at 3 pm')


# In[ ]:


Kolmogorow_Smirnov_Test(weather_data_partly_preprocessed,'Pressure9am')


# In[ ]:


Kolmogorow_Smirnov_Test(weather_data_partly_preprocessed,'Pressure3pm')


# In[ ]:


Q_Q_plot(weather_data_partly_preprocessed, 'Pressure9am')


# In[ ]:


Q_Q_plot(weather_data_partly_preprocessed, 'Pressure3pm')


# ## <span style='color: #3F2E3E;'>Sunshine - continuous - ratio</span>
# #### <span style='color: #A78295;'>Bright sunshine during a particular day. (hours) </span>

# In[23]:


plot_analysis_numerical_variable(weather_data_partly_preprocessed,'Sunshine','Sunshine (h)','Bright sunshine during a day')


# In[24]:


weather_data_sunshine = weather_data_partly_preprocessed.copy()
# weather_data_sunshine = weather_data_sunshine.dropna(subset='Sunshine')


# In[ ]:


#weather_data_partly_preprocessed[weather_data_partly_preprocessed['Sunshine'].isnull()]['State/Province'].value_counts()


# In[ ]:


# Fill the gaps -> KDE kernel density estimation for non-normally distributed data
# I also notice that in windgustspeed we have only integers, the CDE is not a continuous line, what may indicate that 
# the data distribution is higly irregular or discrete and the KDE approach may not be the most appropriate there. 
# This is because the KDE might not accurately capture the distribution, and sampling from it might lead to unrealistic imputed values.
# Fortunately, in our both examples we have continuous CDE


# In[25]:


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


# In[26]:


weather_data_sunshine = random_imputation(weather_data_sunshine, 'Sunshine')


# In[27]:


plot_analysis_numerical_variable(weather_data_sunshine,'Sunshine','Sunshine (h)','Bright sunshine during a day')


# In[28]:


weather_data_partly_preprocessed['Sunshine'].describe()


# In[29]:


#weather_data_sunshine.isnull().sum()
weather_data_sunshine['Sunshine'].describe()


# In[ ]:


#should i generate only outputs from the range that i have or can i somehowe modify the range? There is a difference in max values
# if i use np.clip to ensure that the imputed values remain within the original range (bounded by the minimum and maximum values of the observed data)
# the results will be catastrophic, different mean std cdf -> i think better to stick to the first option


# ## <span style='color: #3F2E3E;'>Evaporation - continuous - ratio</span>
# #### <span style='color: #A78295;'>Evaporation during a particular day. (millimeters)</span>

# In[30]:


plot_analysis_numerical_variable(weather_data_sunshine,'Evaporation','Evaporation (mm)','Evaporation during a particular day.')


# In[31]:


weather_data_evaporation = weather_data_sunshine.copy()


# In[32]:


weather_data_evaporation = random_imputation(weather_data_evaporation, 'Evaporation')


# In[33]:


plot_analysis_numerical_variable(weather_data_evaporation,'Evaporation','Evaporation (mm)','Evaporation during a particular day.')


# In[34]:


weather_data_sunshine['Evaporation'].describe()


# In[35]:


weather_data_evaporation['Evaporation'].describe()


# In[36]:


weather_data_preprocessed = weather_data_evaporation.copy()

