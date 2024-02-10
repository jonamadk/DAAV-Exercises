import pandas as pd



incomeDataframe = pd.read_csv('Income Dirty Data.csv', index_col=0)


total_NaN_count_across_col = incomeDataframe.isna().sum()
total_NaN_count = total_NaN_count_across_col.sum()
percentage_of_NaN_values = total_NaN_count/(len(incomeDataframe)*len(incomeDataframe.columns))


# Calculate the number of observations
total_observations = len(incomeDataframe)
#Calculate the number if incomplete observations
incomplete_observation_count = incomeDataframe[incomeDataframe.isna().any(axis=1)]
# Calculate the number of complete observations
complete_observation_count = len(incomeDataframe.dropna())
# Calculate the percentage of complete observations
percentage_complete = (complete_observation_count / total_observations) * 100


print(f"Total observations: {total_observations}")
print(f"Complete observations: {complete_observation_count}")
print(f"Percentage of complete observations: {percentage_complete:.2f}%")



#2. Checking with rules
non_violating_rows = incomeDataframe[(incomeDataframe['age']>=18) & (incomeDataframe['tax (15%)']== incomeDataframe['income']*.15) & (incomeDataframe['income']>0)] 
print(f"Percentage of the data has no errors (i.e., rows that don’t violate the above rules):{(non_violating_rows.shape[0]/incomeDataframe.shape[0])*100:.2f}%")


