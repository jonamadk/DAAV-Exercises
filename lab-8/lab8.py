import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats.contingency import  margins




americanDataframe = pd.read_csv("Americandata.csv")

print("\n 1. Reading Dataframe:")
print(americanDataframe)

stat, p, dof, expected = chi2_contingency(americanDataframe.iloc[:,1:].values)

print("\n 2. The chi2 results\n")
print("dof=%d" % dof)
print("p=%s" %p)
print("stat=%s" % stat)
print("\nExpected Values")
print(expected)



def stdres(observed, expected):
    n = observed.sum()
    rsum, csum = margins(observed)
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum)/ n**3

    return (observed - expected) / np.sqrt(v)



print("\n3. The standardized residuals")
americanDataframeResiduals = stdres(americanDataframe.iloc[:,1:].values, expected)
print(americanDataframeResiduals)

