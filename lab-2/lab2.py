import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





def descriptive_stat():

    """
        _input_: CSV file
        
    
    """
    dataframe = pd.read_csv('Salary Data - Ex2.csv')
    salary, education, prestige = dataframe['salary'], dataframe['education'], dataframe['prestige']
    
    #Salary Descriptive insights
    salary_percentiles = np.percentile(salary, [0, 25, 50, 75,100], interpolation= "linear")
    print("Salary Minimum Value",salary_percentiles[0])
    print("Salary 1st Quartile", salary_percentiles[1])
    print("Salary Median", salary_percentiles[2])
    print("Salary Mean", np.mean(salary))
    print("Salary 3rd Quartile", salary_percentiles[3])
    print("Salary Maximum value",salary_percentiles[4])
    print("\n")

    
    # Education Descriptive insights
    education_percentiles = np.percentile(education, [0, 25, 50, 75,100], interpolation= "linear")
    print("education Minimum Value",education_percentiles[0])
    print("education 1st Quartile", education_percentiles[1])
    print("education Median", education_percentiles[2])
    print("education Mean", np.mean(education))
    print("education 3rd Quartile", education_percentiles[3])
    print("education Maximum value",education_percentiles[4])
    print("\n")
    
    
    # Prestige Descriptive insights
    prestige_percentiles = np.percentile(prestige, [0, 25, 50, 75,100], interpolation= "linear")
    print("prestige Minimum Value",prestige_percentiles[0])
    print("prestige 1st Quartile", prestige_percentiles[1])
    print("prestige Median", prestige_percentiles[2])
    print("prestige Mean", np.mean(prestige))
    print("prestige 3rd Quartile", prestige_percentiles[3])
    print("prestige Maximum value",prestige_percentiles[4])
    print("\n")

    
    #Calculate Variance and Standard Deviation
    print("Salary Variance", salary.var())
    print("Salary Standard deviation", salary.std())
    
    #Histogram and Scatter Plot Visualization
    dataframe.hist(column='prestige', edgecolor='black')
    plt.title("Presitge Distribution")
    dataframe.hist(column='education', edgecolor='black')
    plt.title("Education Distribution")
    dataframe.plot.scatter("salary", "education")
    plt.title("Salary vs Education")
    dataframe.plot.scatter("education", "prestige")
    plt.title("Education vs Prestige")
    plt.show()
    
    
    
descriptive_stat()



    
    
    