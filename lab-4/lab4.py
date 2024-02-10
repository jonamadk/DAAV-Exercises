import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder , StandardScaler 
from sklearn.impute import KNNImputer

incomeDataframe = pd.read_csv('Income Dirty Data.csv', index_col=0)


total_NaN_count_across_col = incomeDataframe.isna().sum()
total_NaN_count = total_NaN_count_across_col.sum()
percentage_of_NaN_values = total_NaN_count/(len(incomeDataframe)*len(incomeDataframe.columns))


# Calculate the number of observations
total_observations = len(incomeDataframe)
#Calculate the number if incomplete observations
incomplete_observation_count = len(incomeDataframe[incomeDataframe.isna().any(axis=1)])
# Calculate the number of complete observations
complete_observation_count = len(incomeDataframe.dropna())
# OR complete_observation_count = total_observations - incomplete_observation_count

# Calculate the percentage of complete observations
percentage_complete = (complete_observation_count / total_observations) * 100


print(f"Total observations: {total_observations}")
print(f"Complete observations: {complete_observation_count}")
print(f"Percentage of complete observations: {percentage_complete:.2f}%\n")



#2. Checking with rules
non_violating_rows = incomeDataframe[(incomeDataframe['age']>=18) & (incomeDataframe['tax (15%)']== incomeDataframe['income']*.15) & (incomeDataframe['income']>0)] 
print(f"Percentage of the data has no errors (i.e., rows that don’t violate the above rules):{(non_violating_rows.shape[0]/incomeDataframe.shape[0])*100:.2f}%\n")


#3. Correcting
incomeDataframe['gender'] = incomeDataframe['gender'].apply(lambda gender: 'Male' if(gender == 'Man' or gender == 'Men') else 'Female' if (gender == "Women" or gender =="Woman") else gender)



incomeDataframe.loc[~(incomeDataframe['age']>0),'age'] = np.NaN
incomeDataframe.loc[~(incomeDataframe['income']>0),'income'] = np.NaN
incomeDataframe.loc[~(incomeDataframe['tax (15%)']>0),'tax (15%)'] = np.NaN


incomeDataframe['income'] = np.where(incomeDataframe['income'].isna() & incomeDataframe['tax (15%)'].notnull(), incomeDataframe['tax (15%)']*(100/15), incomeDataframe['income'])
incomeDataframe['tax (15%)'] = np.where(incomeDataframe['tax (15%)'].isna() & incomeDataframe['income'].notnull(), incomeDataframe['income']*.15, incomeDataframe['tax (15%)'])



print(incomeDataframe.describe(pd.set_option('display.float_format', lambda x: '%.4f' % x)).rename(index={'25%': '1st Qua', '50%': 'Median', '75%': '3rd Qua'}),"\n")
print("Missing Values Count Before Imputation ")
print(incomeDataframe.isna().sum(),"\n")


label_encoder = LabelEncoder()
standard_scaler = StandardScaler()



categorical_columns = incomeDataframe.select_dtypes(include=['object']).columns

#Performing LabelEncoding on categorical data
for column in categorical_columns:
    incomeDataframe[column] = label_encoder.fit_transform(incomeDataframe[column])


# Scaling/Normalizing the features
features = [incomeDataframe.columns.tolist()]
for column in features:
    incomeDataframe[column] = standard_scaler.fit_transform(incomeDataframe[column])



#Performing imputation
knn_imputer = KNNImputer()
complete_income_df= knn_imputer.fit_transform(incomeDataframe)
finalIncomeDataset = pd.DataFrame(data= complete_income_df, columns = incomeDataframe.columns.tolist())
finalIncomeDataset.index.name = "ID"



print(finalIncomeDataset.describe().rename(index={'25%': '1st Qua', '50%': 'Median', '75%': '3rd Qua'}),"\n")
print("Missing Values Count After Imputation ")
print(finalIncomeDataset.describe().isna().sum())



