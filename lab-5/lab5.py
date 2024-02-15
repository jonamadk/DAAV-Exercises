import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



album_dataframe = pd.read_csv("Lab Album Sales.csv")


colors = np.array([0.1,0.6,0.8])
area = np.pi*3


plt.scatter(album_dataframe.totalsales, album_dataframe.AdvertBudget, s=area, c=colors,alpha=0.5)
plt.title('Sales Vs. Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
# plt.show()

plt.scatter(album_dataframe.totalsales, album_dataframe.AirplayTimes, s=area, c=colors,alpha=0.5)
plt.title('Sales Vs. Airplay')
plt.xlabel('Sales')
plt.ylabel('Airplay')
# plt.show()

plt.scatter(album_dataframe.totalsales, album_dataframe.AttractivenessScore, s=area, c=colors,alpha=0.5)
plt.title('Sales Vs. Attractiveness')
plt.xlabel('Sales')
plt.ylabel('Attractiveness')
# plt.show()


lm1 = smf.ols(formula='totalsales ~ AdvertBudget', data=album_dataframe).fit()
print("\n Linear Regression p-value and summary")
print(lm1.pvalues.to_string())
print(lm1.summary())



print(f"\n Model parameters:\n {lm1.params.to_string()}")
print("\nEstimation for the $135,000 in Advert")
print(lm1.params[0]+ lm1.params[1]*135000)



lm2= smf.ols(formula='totalsales ~ AdvertBudget + AirplayTimes + AttractivenessScore',data=album_dataframe).fit()
print("Multi-Variable linear Regression p-values and Summary")
print(lm2.pvalues.to_string())
print(lm2.summary())
