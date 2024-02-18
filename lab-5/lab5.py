#Package Import Section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


#Reading dataset
album_dataframe = pd.read_csv("Lab Album Sales.csv")

#Defing color code array
colors = np.array([0.1,0.6,0.8])
area = np.pi*3

# Scatterplot for Sales Vs. Advertising
plt.scatter(album_dataframe.totalsales, album_dataframe.AdvertBudget, s=area, color=colors,alpha=0.5)
plt.title('Sales Vs. Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.show()

#Scatterplot for Sales Vs. Airplay
plt.scatter(album_dataframe.totalsales, album_dataframe.AirplayTimes, s=area, color=colors,alpha=0.5)
plt.title('Sales Vs. Airplay')
plt.xlabel('Sales')
plt.ylabel('Airplay')
plt.show()

#Scatterplot for Sales Vs. Attractiveness
plt.scatter(album_dataframe.totalsales, album_dataframe.AttractivenessScore, s=area, color=colors,alpha=0.5)
plt.title('Sales Vs. Attractiveness')
plt.xlabel('Sales')
plt.ylabel('Attractiveness')
plt.show()

#Constructing Linear Model for  sales prediction based on advertisement Budget
linear_model_1 = smf.ols(formula='totalsales ~ AdvertBudget', data=album_dataframe).fit()
print("\n Linear Regression p-value and summary")
print(linear_model_1.pvalues.to_string())
print(linear_model_1.summary())


# Prediction on the  no. of record sales of ablum based on some advertisement Budget
print(f"\nLinear Regression Model parameters:\n {linear_model_1.params.to_string()}")
print("\nEstimation for the $135,000 in Advert")
print(linear_model_1.params.iloc[0]+ linear_model_1.params.iloc[1]*135000,"\n")


#Constructing Multiple Linear Regression model for Sales based on more than one independent variable.
linear_model_2= smf.ols(formula='totalsales ~ AdvertBudget + AirplayTimes + AttractivenessScore',data=album_dataframe).fit()
print("Multi-Variable linear Regression p-values and Summary")
print(linear_model_2.pvalues.to_string())
print(linear_model_2.summary())
