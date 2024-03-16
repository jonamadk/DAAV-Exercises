import pandas as pd
from apyori import apriori


titanic_dataframe = pd.read_csv("TitanicData.csv")

#Oberving the dataframe/dataset
print(titanic_dataframe.head(7),"\n") #The pandas framework detected the index column and auto-fitted the columns


#Checking unique values of the columns
for column in titanic_dataframe.columns.tolist():
    print(f"Unique value for {column} :{titanic_dataframe[column].unique()}")
    

#Performing apriori association rules
association_rules = apriori(titanic_dataframe.values, min_support = 0.005, min_confidence =0.8, min_length=2)
association_results = list(association_rules)


#Finding optimal filtered results that has Survival chance - Yes
optimal_filtered_results = [entry for result in association_results \
                            for entry in result.ordered_statistics \
                            if entry.items_add == frozenset({"Yes"})]

print(f"\nNumber of unique rules is {len(optimal_filtered_results)}\n")

#Sortinf based on Lift value
sorted_results = sorted(optimal_filtered_results, key=lambda x: x.lift, reverse= True)

print("Final Result:\n")
for result in sorted_results:
    print(str(result))