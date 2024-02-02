import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as stat


dataframe = pd.read_csv('Salary Data - Ex3.csv')

salary, education, prestige = dataframe['salary'], dataframe['education'], dataframe['prestige']


#Covariance for salary and education.
print(f"Covariance of Salary and Education is {salary.cov(education)}")
print(f"Covariance of Salary and Prestige is {salary.cov(prestige)}")
print(f"Covariance of Education and Prestige is {education.cov(prestige)}")


# Pearson’s correlation coefficients and the p-values for salary and education.

salary_education_r_value, salary_education_p_value = stat.pearsonr(salary, education)
print(f"Pearson correlation coefficients is {salary_education_r_value} and p-values is {salary_education_p_value}" )

salary_prestige_r_value, salary_prestige_p_value = stat.pearsonr(salary, prestige)
print(f"Pearson correlation coefficients is {salary_prestige_r_value} and p-values is {salary_prestige_p_value}" )

education_prestige_r_value, education_prestige_p_value = stat.pearsonr(education, prestige)
print(f"Pearson correlation coefficients is {education_prestige_r_value} and p-values is {education_prestige_p_value}" )