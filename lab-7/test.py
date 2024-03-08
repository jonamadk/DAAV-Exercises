
import pandas as pd
from apyori import apriori

titanicDF = pd.read_csv("TitanicData.csv")

#print the dataframe

print("Checking and Reading the Data: ")
print(titanicDF.head())

for col in titanicDF.columns: 
    print("\nUnique values for {0}:".format(col))
    for val in titanicDF[col].unique():
        print(val)

#collecting the inferred rules in a dataframe

association_rules = apriori(titanicDF.values, min_support=0.005, min_confidence=0.8, min_length=2)
        
association_results = list(association_rules)

filteredResults = []
#filtering our results to just rules that rhs is survived

for result in association_results:
    for entry in result.ordered_statistics:
        if entry.items_add == frozenset({'Yes'}):
            filteredResults.append(entry)

print("\nNumber of Rules: {0}\n".format(len(filteredResults)))


    
s