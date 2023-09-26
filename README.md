# Data-science

# Project 1 - Australian rainfall prediction

Goal: To predict the next-day rain based on other atmospheric features

Dataset: This dataset comprises a decade of daily weather observations from multiple locations across Australia.

Source: https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data

What is included in the project? 
1. The map of Australia with data points
2. Preprocessing - separate files for categorical and numerical data with EDA 1D + merged 
3. EDA 2D 
4. ML - logistic regression 

Data Description:
Location - Name of the city from Australia.

MinTemp/MaxTemp - The minimum/maximum temperature during a particular day. (degree Celsius)

Rainfall - Rainfall during a particular day. (millimeters)

Evaporation - Evaporation during a particular day. (millimeters)

Sunshine - Bright sunshine during a particular day. (hours)

WindGusDir - The direction of the strongest gust during a particular day. (16 compass points)

WindGuSpeed - Speed of strongest gust during a particular day. (kilometers per hour)

WindDir9am / WindDir3pm - The direction of the wind for 10 min prior to 9 am. / 3pm. (compass points)

WindSpeed9am / WindSpeed3pm - Speed of the wind for 10 min prior to 9 am. / 3pm. (kilometers per hour)

Humidity9am / Humidity3pm - The humidity of the wind at 9 am. / 3pm. (percent)

Pressure9am / Pressure3pm - Atmospheric pressure at 9 am. / 3pm. (hectopascals)

Cloud9am / Cloud3pm - Cloud-obscured portions of the sky at 9 am. / 3pm.(eighths)

Temp9am / Temp3pm - The temperature at 9 am. / 3pm.(degree Celsius)

RainToday - If today is rainy then ‘Yes’. If today is not rainy then ‘No’.

RainTomorrow - If tomorrow is rainy then 1 (Yes). If tomorrow is not rainy then 0 (No).

The variables below were not presented in original dataset. However, I thought it would be wise to add regions and coordinates in order to plot the cities on the map (which I prepared in Tableau). Always better to see where the places we are taking about are.
State/Province - State/Province of the locations in Australia

Longitute/Latitude - Coordinates of mentioned cities

It's a pity that there is no information about the date, however when we plot some variables I'm sure we will receive some seasonal trends in the data for example in min max temperature.

Why do we measure "clouds", "wind direction","humidity","temperature" at 9am and 3pm?

It is also important to become acquainted with Australian climate especially in respect of rainfall before we will immerse in the analysis of our dataset. Here are some key aspects:

Australia is located in the Southern Hemisphere and is surrounded by oceans (Southern Ocean,Pacific Ocean, Indian Ocean) and seas (Timor Sea,Arafura Sea,Coral Sea, Tasman Sea).

Dry and arid regions particularly in the central and western regions. These areas experience hot and dry conditions for much of the year, with limited rainfall. They are characterized by vast deserts, such as the Simpson Desert and the Great Victoria Desert.

Northern parts of Australia have tropical / subtropical climate - wet/dry seasons; monsoon seasons (wet) - from November to April,heavy rainfall,tropical cyclones; dry seasons - from May to October, lower humidity, clear skies

Southern / Southeastern regions have temparate climate - mild winters, moderate summers, rainfall throughout the year

Southern / Southwestern parts have mediterranean climate - mild, wet winters and hot,dry summers

Apline climate in southeastern regions (particularly the Australian Alps) - cold winters,snowfall,cool summers, higher rainfall due to orographic effects(rainfall caused by the lifting of moist air over mountains)

Coastale areas have mild temperatures and high humidity

Australia is prone to natural disasters, including tropical cyclones in the north and bushfires in various parts of the country.

To conclude: Australia's climate is highly diverse and can be broadly classified into several distinct regions based on precipitation patterns







