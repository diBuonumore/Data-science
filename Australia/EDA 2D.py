#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install folium
# !pip install geopandas


# In[1]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import folium
from folium import plugins


# In[2]:


pd.set_option('display.max_columns', None)
weather_data = pd.read_csv("Preprocessing completed-Australian Rainfall.csv")
weather_data.head(5)


# In[ ]:


# weather_data.columns


# In[ ]:


# weather_data.isnull().sum()


# ## Correlation - the dependency between numerical values

# In[3]:


columns_to_exclude = ['Location','State/Province','WindGustDir','WindDir9am','WindDir3pm','Cloud9am','Cloud3pm','RainTomorrow','RainToday']
len(columns_to_exclude)
selected_columns = [col for col in weather_data.columns if col not in columns_to_exclude]
numeric_df = weather_data[selected_columns]


# In[4]:


correlation_matrix = numeric_df.corr(numeric_only=True)
correlation_matrix

def color_negative_red(val):
    color = 'red' if val > 0.7 else 'black'
    return 'color: %s' % color

correlation_matrix.style.applymap(color_negative_red)


# # To measure the dependency between a categorical variable and a numerical variable, we may use varoius methods:
# 1. **ANOVA (Analysis of Variance)**-  for differences in means between multiple groups of a categorical variable with respect to a numerical variable.
# It assesses whether there is a statistically significant difference in the means of the numerical variable across the different categories.
# However, ANOVA assumes that the numerical variable follows a normal distribution within each group. 
# 
# 2. **Kruskal-Wallis Test**: is a non-parametric test that can be used when the assumption of normality is violated (is suitable when the numerical variable is not normally distributed).
# It tests whether the median of the numerical variable differs significantly across the categories of the categorical variable.
# 
# 4. **Mann-Whitney U Test** (Wilcoxon rank-sum test): non-parametric test,  assesses whether there are significant differences in the distribution of the numerical variable between the two groups defined by the binary categorical variable. This test however does not provide information about the direction or magnitude of the difference between the two groups. It only assesses whether there is a significant difference in the distributions.
#                                                                                                                 
# 3. **Point-Biserial Correlation / Biserial Correlation**: it is a measure of association between a binary (dichotomous - 2 groups only) variable and a continuous variable. It is similar to the standard Pearson correlation. This correlation coefficient measures the strenght and direction of the <u>linear relationship</u> between variables.It ranges from -1 to 1, where -1 and 1 indicates perfect linear relationship, 0 no linear relationship. 
# 
# There are also other options but those are the most popular.  
#                                                                                                                 
# I decided to try both Point-Biserial Correlation and Mann-Whitney U Test. The second one, I also wanted to complete with the plots (violin plots and swarm plot) but swarm plot did not load (probably too much data) and because of it, the differences in the distributions we will see only on violin plot. 

# ## Point-Biserial Correlation 

# In[5]:


from scipy.stats import pointbiserialr

def point_biserial(binary_variable,continuous_variable):
    correlation_coefficient, p_value = pointbiserialr(binary_variable, continuous_variable)

#     print("Point-Biserial Correlation:")
    print("Correlation Coefficient:", correlation_coefficient)
    print("P-value:", p_value,"\n")


# In[ ]:


# numeric_df.head(5)


# In[6]:


# the correlation coef -> [-1,1], ~ -1,1 strong association, ~ 0 weak
# p-value - measure of the statistical significance of the correlation coef. It represents the probability of obtaining 
# the observed correlation coef by chance (typically 5%). If p-value < 0.05 correlation coef is statistically significant
# and the observed association is unlikely to be due to random chance.

print("Point-Biserial Correlation between RainTomorrow and other numerical variables\n")
for num in numeric_df: 
    print(num)
    point_biserial(weather_data['RainTomorrow'],numeric_df[num])
# all p-values <0.05 -> correlation coefs are statistically significant
# coef: rather they indicate weak association, some of them are positive some negative


# In[ ]:


# weather_data_preprocessed.info()


# ## Mann-Whitney U Test

# Procedure: 
# 
#     1. H_0 - there is no difference in the distribution of the numerical variable between the two groups defined by the categorical variable 
#     2. H_1 - there is a significant difference -|| - 
#     3. Rank the data: 
#         - order the numerical variable from the lowest to highest 
#         - assign ranks to the data points starting from 1 for the lowest value and increasing by 1 for each subsequent value
#         - ties (repeated values) receive the average rank of the tied ranks. ( example 1.5 1.5 3 4 5 ) 
#     4. Calculate the U-statistic (there are formulas for small and large number of samples) and p-value
#     5. Interpretation: if p-value < 0.05 reject the null hypothesis, if >= 0.05 fail to reject H_0

# In[7]:


from scipy.stats import mannwhitneyu

def mann_whitney_u_test(categorical_variable,numerical_variable):
    statistic, p_value = mannwhitneyu(numerical_variable[categorical_variable == 1],
                                  numerical_variable[categorical_variable == 0])

    alpha = 0.05
    print("U-statistic:", statistic)
    print("P-value:", p_value)
    if p_value < alpha:
        print("Reject the null hypothesis. There are significant differences between the groups.\n")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the groups.\n")


# In[8]:


print("The Mann-Whitney U test between RainTomorrow and other numerical variables\n")
for num in numeric_df: 
    print(num)
    mann_whitney_u_test(weather_data['RainTomorrow'],numeric_df[num])

# In summary, for all the numerical variables tested, the Mann-Whitney U test rejects the null hypothesis 
# and indicates that there are significant differences in the distributions of each numerical variable between RainTomorrow groups


# ## Violin plot
# It is a hybrid of a box plot and a kernel density plot. It shows the distribution of numerical data. 
# 
# Components: 
# 
#     - the white dot - median
#     - the box plot in the middle
#     - the thick grey bar in the center - interquartile range (50% of observations -> Q3-Q1)
#     - kernel density plot on the both sides of the violin - shows the probability density of data points at different values
#     width = freqency; the wider range, the more frequent value (higher probability)
#     - outliers - data points that fall outside the whiskers of the box plot
#     
# From the violin plot, we can find out for example how many values clustered around the median, min, max or we may compare the medians for different categories.
# 
# If the plot is extremely skinny on each end and wide in the middle -> observations are mainly concentrated around the median.
# 
# For many categories we can swap the axes (horizontal violin).

# In[ ]:


# sns.set(style="whitegrid", palette="pastel", color_codes=True)
# sns.violinplot(x="feed", y="weight", hue="sex", data=df,
#                palette={"male": "b", "female": "y"})
# plt.legend(loc='upper left')
# https://mode.com/blog/violin-plot-examples/


# In[9]:


def violin_plot(df,categorical_variable,numerical_variable):
    custom_palette = ["#DBDFEA", "#8294C4"]
    sns.violinplot(x=categorical_variable, y=numerical_variable, data=df,palette=custom_palette)
    sns.despine(left=True)

#     plt.title("Violin Plot",fontweight='bold')
    plt.xlabel(categorical_variable)
    plt.ylabel(numerical_variable)
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    plt.show()


# In[ ]:


# for col in numeric_df.columns:
#     violin_plot(weather_data,'RainTomorrow',col)
#observations: mostly values are concentrated around the median,in some variables there are distincive outliers like in rainfall or evaporation


# In[10]:


violin_plot(weather_data,'RainTomorrow','Rainfall')


# In[ ]:


#sns.swarmplot(x='categorical_var', y='numerical_var', data=data)


# ## The dependency between RainTomorrow and other categorical variables

# ### The chi-square test - categorical vs. categorical variable
# The Chi-square test is used to determine if there is a significant association between two categorical variables. 
# Process for conducting a Chi-square test:
# 1. Formulate the null and alternative hypotheses:
# Null hypothesis: There is no association between the two categorical variables.
# Alternative hypothesis: There is a significant association - || - 
# 2. Set the significance level (alpha): Choose a significance level (commonly 0.05) that represents the threshold for determining statistical significance. The significance level determines how strong the evidence against the null hypothesis must be before we reject it.
# 3. Create a contingency table (cross-tabulation)- a table that shows the observed frequencies for each combination of the two categorical variables.
# 4. Compute the Chi-square test statistic - you calculate the difference between the observed and expected frequencies and then you sum up them.
# 5. Find p-value
# 6. Comparison: If the p-value is < 0.05, reject the null hypothesis - there is a significant association between the variables. If The p-value >= 0.05 reject alternative hypothesis - there is no significant association.
# 7. Chi-square statistic: The bigger difference, the stronger association between the variables. When the test statistic is large, it suggests that there is a significant discrepancy between the observed and expected frequencies, indicating that the variables are dependent.
# 
# #### I added also Cramer's V, which is an extension of the chi-square test and provides a standardized measure of association.
# Cramer's V ranges from 0 to 1, where 0 indicates no association between the categorical variables, and 1 indicates a perfect association. The formula for Cramer's V is:
# 
# $$
# V = \sqrt{\frac{\chi^2}{(n \cdot min (k-1,r-1)}}
# $$
# 
# Where:
# 
# $\chi^2$  is the chi-square statistic obtained from the chi-square test of independence between the two categorical variables.
# 
# n is the total number of observations in the contingency table.
# 
# k is the number of rows in the contingency table.
# 
# r is the number of columns in the contingency table.

# In[11]:


from scipy.stats import chi2_contingency
def chi_square_test(categorical_var_1, categorical_var_2):
    crosstab = pd.crosstab(categorical_var_1, categorical_var_2)
    chi2, p, _, _ = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    # crosstab.sum() - the sum of elements along each column of the contingency table
    # crosstab.sum().sum() - the sum of all elements in the 1-dimensional array obtained from the previous step
    num_rows = crosstab.shape[0]
    num_cols = crosstab.shape[1]
    cramer_v = np.sqrt(chi2 / (n * min(num_rows - 1, num_cols - 1)))
    return chi2, p, cramer_v

def chi_square_contigency(tab_x, tab_x_str, var,sort=1):
    df = pd.DataFrame(columns=['Variable', 'Chi-square', 'P-value',"Cramer's V"])
    for x, x_str in zip(tab_x, tab_x_str):
        chi2, p, cramer_v = chi_square_test(var, x)
        result = pd.DataFrame({'Variable': [x_str], 'Chi-square': [round(chi2,2)], 'P-value': [p], "Cramer's V":[round(cramer_v,2)]})
        df = pd.concat([df, result], ignore_index=True)
        if sort:
            df = df.sort_values("Cramer's V",ascending=False)
    return df


# In[12]:


# Problem 1) Cloud's type is by default float, so I need to change it:
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

cloud_9am = weather_data['Cloud9am'].map(cloud_categories).astype('category')
cloud_3pm =  weather_data['Cloud3pm'].map(cloud_categories).astype('category')

#other categorical variables
location = weather_data['Location']
state = weather_data['State/Province']
wind_gust = weather_data['WindGustDir']
wind_9am = weather_data['WindDir9am']
wind_3pm = weather_data['WindDir3pm']
rain_today = weather_data['RainToday']

# 'RainTomorrow' - 0 if will be raining 1 if not - change to an object
rain_tomorrow = weather_data['RainTomorrow'].map({0:'No',1:'Yes'})

categorical_tab = np.array([location,state,wind_gust,wind_9am,wind_3pm,cloud_9am,cloud_3pm,rain_today,rain_tomorrow])
categorical_tab_str = np.array(['location','state','wind_gust','wind_9am','wind_3pm','cloud_9am','cloud_3pm','rain_today','rain_tomorrow'])


# In[ ]:


#"correlation does not imply causation". It means that just because two variables show a statistical relationship (correlation) does not necessarily mean that one variable causes the other to change.
# pd.set_option('display.max_columns', None)
# pd.crosstab(cloud_3pm,rain_tomorrow)
# pd.crosstab(cloud_9am,rain_tomorrow)
# pd.crosstab(rain_tomorrow,location)
# pd.crosstab(rain_tomorrow,state)
# pd.crosstab(rain_tomorrow,wind_gust)
# pd.crosstab(rain_tomorrow,wind_9am)
# pd.crosstab(rain_tomorrow,wind_3pm)
# pd.crosstab(rain_tomorrow,rain_today)


# In[13]:


chi_square_test(rain_tomorrow, state)


# In[14]:


chi_square_contigency(categorical_tab,categorical_tab_str,rain_tomorrow,0)
chi_square_contigency(categorical_tab,categorical_tab_str,rain_tomorrow)


# **In summary**
# The large chi-squares and small p-values (close to 0) indicate strong evidence against the null hypothesis and prove that there is a significant association between rain_tomorrow and other variables. Moreover,The strength of association varies, with some variables exhibiting stronger associations (Cramer's V closer to 1) and others having weaker associations (Cramer's V closer to 0). 

# ## How are the other categorical variables related to each other?

# In[15]:


cramers_v_table = pd.DataFrame()
for index, col in enumerate(categorical_tab):
    cramers_v_table[index]=chi_square_contigency(categorical_tab,categorical_tab_str,col,0)["Cramer's V"]
cramers_v_table


# In[16]:


keys = np.arange(9)
values = categorical_tab_str
new_names = {}
for i in range(len(keys)):
    new_names[keys[i]]=values[i]
# new_names


# In[17]:


cramers_v_table.rename(columns=new_names, inplace=True)
cramers_v_table.rename(index=new_names, inplace=True)
cramers_v_table

def color_negative_red(val):
    color = 'red' if val > 0.5 else 'black'
    return 'color: %s' % color

cramers_v_table.style.applymap(color_negative_red)


# In[ ]:


#print("\033[31mI did not know \033[32;1mthat I can so much change the style \033[33mof the text \033[0min python")


# In[ ]:


# https://colorhunt.co/palettes/vintage


# ## Plots 2D  
# Ideas: 
# 
#     1. Map (json file) with points (long,lat) - heatmap,how many points are concentrated in certain areas 
#     2. Identification of regional weather patterns: on the map, heatmap with variables like rainfall,windspeedn,pressure 
#     3. Avg min/max temp for specific locations + Avg temp9am/temp3pm
#     
# Facts:
# 
#     1. the lower humidity, the higher evaporation 
#     2. when sunshine increases, it generally leads to an increase in evaporation.
#     The relationship in a nutshell: 
#     More Sunshine → Higher Surface Temperature → Increased Kinetic Energy of Water Molecules → Enhanced Evaporation

# ## Maps - density of observations

# In[18]:


m = folium.Map(location=[-26.4390917,133.281323],zoom_start=4)
folium.TileLayer('OpenStreetMap').add_to(m)
# folium.TileLayer('Stamen Water color').add_to(m)
# folium.TileLayer('cartodbpositron').add_to(m)
# folium.TileLayer('CartoDB Positron').add_to(m)

plugins.HeatMap(
    weather_data[['Latitude','Longitude']],
    name='Heatmap'
).add_to(m)

m


# ## Heatmap on the map

# In[19]:


shape = gpd.read_file("Shapefiles\STE_2021_AUST_GDA2020.shp")
# shape


# In[20]:


shape.shape


# In[21]:


sorted(shape['STE_NAME21'].unique())


# In[22]:


sorted(weather_data['State/Province'].unique())


# In[23]:


for index, state in enumerate(shape['STE_NAME21']):
        shape.loc[index, 'STE_NAME21'] = f" {state}"


# In[ ]:


sorted(shape['STE_NAME21'].unique())


# In[ ]:


shape = shape[shape['STE_NAME21']!=' Outside Australia']


# In[ ]:


shape.loc[8, 'STE_NAME21'] = ' Norfolk Island '


# In[ ]:


shape.rename(columns ={'STE_NAME21':'State/Province'},inplace=True)


# In[ ]:


shape


# In[ ]:


# map = shape.merge(weather_data, on='State/Province')
# map.head(5)


# In[ ]:


# fig, ax = plt.subplots(figsize=(8, 8))
# map['geometry'] = map['geometry'].apply(lambda geom: geom.simplify(tolerance=0.001))
# map.plot(column='Rainfall',cmap='Reds',linewidth=0.4,ax=ax,edgecolor=".4")


# ## Heatmaps

# In[25]:


heatmap_data = weather_data.pivot_table(index='State/Province', values=['WindSpeed3pm','WindSpeed9am'])

#Reorder columns
heatmap_data = heatmap_data[['WindSpeed9am', 'WindSpeed3pm']] 

fig, ax = plt.subplots(figsize=(10, 8))

# Create the heatmap
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=0.5, ax=ax,cbar_kws={'label': 'Wind Speed [km/h]'})


plt.show()


# In[26]:


heatmap_data = weather_data.pivot_table(index='Location', values=['Humidity9am','Humidity3pm'])
heatmap_data = heatmap_data[['Humidity9am', 'Humidity3pm']] 

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=0.5, ax=ax,cbar_kws={'label': 'Humidity [%]'})

plt.show()


# In[27]:


heatmap_data = weather_data.pivot_table(index='Location', values=['Pressure9am','Pressure3pm'])
heatmap_data = heatmap_data[['Pressure9am', 'Pressure3pm']] 

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=0.5, ax=ax,cbar_kws={'label': 'Pressure [hPa]'})

plt.show()


# ## Thesis: the higher humidity, the lower evaporation 
# Let's verificate it

# In[28]:


fig,ax = plt.subplots(figsize=(10,8))

sns.lineplot(data=weather_data,  y='Evaporation',x='Humidity9am',marker='o', color='#765827', ax=ax,label='Humidity9am',markersize=4.5)
sns.lineplot(data=weather_data,  y='Evaporation',x='Humidity3pm', marker='o', color='#C8AE7D', ax=ax,label='Humidity3pm',markersize=4.5)

ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.title('Humidity vs. Evaporation')
ax.set_ylabel('Avg Evaporation [mm]')
ax.set_xlabel('Humidity [%]')

plt.show()

# if i swap x with y, i get for single value of evaporation multiple data points of humidity
# the area around the line - ci: confidence interval of the mean 95%
# CI - is a range of values that gives you an estimate of where the true population parameter
# (such as a mean, proportion, or other statistic) is likely to fall, based on a sample from that population and a specified level of confidence.
# It is hard to confirm on 100% our thesis, however overall, a negative correlation is visible. Nevertheless there are many 
# amibuous segments, i.e. evaporation decreases at times, then slightly increases and decreases again. Similarly with humidity


# ## Thesis: When sunshine increases, it generally leads to an increase in evaporation.

# In[29]:


sunshine_evaporation = weather_data.groupby('Sunshine')['Evaporation'].mean()
# type(sunshine_evaporation)


# In[ ]:


# pd.set_option('display.max_rows',None)
# sunshine_evaporation
# weather_data[weather_data['Sunshine'] == 14.6]['Evaporation']


# In[30]:


fig,ax = plt.subplots(figsize=(10,8))

sns.lineplot(data=sunshine_evaporation, marker = "o", color="#99A98F",ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis = 'y',linestyle='-',alpha=0.7)

plt.title('Sunshine vs. Evaporation')
ax.set_ylabel('Avg Evaporation [mm]')
ax.set_xlabel('Avg Sunshine [h]')

plt.show()      
# Plot is rather ambiguous, there are many local minima and maxima, the data is very diverse. We cannot explicitly say that if 
# ours of sunshine increase, evaporation too. 


# ## Rainfall and Max/Min Temperature

# In[31]:


fig,ax1 = plt.subplots(figsize=(10,8))

sns.barplot(data=weather_data, x='Location', y='Rainfall', color='#8EA7E9', ax=ax1, errorbar=None, width=0.8)
ax1.set_ylabel('Avg rainfall [mm]')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

# Create a line plot for Temperature on the right y-axis
ax2 = ax1.twinx()
sns.lineplot(data=weather_data, x='Location', y='MaxTemp', marker='o', color='#FF6969', ax=ax2,label='Max Temp')
sns.lineplot(data=weather_data, x='Location', y='MinTemp', marker='o', color='#606C5D', ax=ax2, label='Min Temp')
ax2.set_ylabel('Avg temperature [°C]')
ax2.set_ylim(0, 40) 

ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.xlabel('Location')
plt.title('Rainfall and Temperature Plot')
plt.xticks(rotation=90)

plt.show()


# ## All temperatures
# 

# In[32]:


fig,ax = plt.subplots(figsize = (12,8))
sns.lineplot(data=weather_data, x='Location',y='MaxTemp',marker='o',ax=ax,label='Max Temp',color="#AC4425")
sns.lineplot(data=weather_data, x='Location',y='Temp3pm',marker='o',ax=ax,label='Temp at 3pm',color="#E6B325")
sns.lineplot(data=weather_data, x='Location',y='Temp9am',marker='o',ax=ax,label='Temp at 9am',color="#6B4F4F")
sns.lineplot(data=weather_data, x='Location',y='MinTemp',marker='o',ax=ax,label='Min Temp',color="#3C6255")
ax.set_ylabel('Avg temperature [°C]')
ax.grid(axis='y', linestyle='--', alpha=0.7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.xlabel('Location')
plt.title(' Comparison of temperatures')
plt.xticks(rotation=90)
plt.show()

