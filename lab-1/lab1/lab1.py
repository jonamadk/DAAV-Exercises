import pandas as pd

__author__ = "Manoj Adhikari"


FILENAME = "Lab 1 Data.csv"



def dataframe_analyzer(FILENAME):
    
    """
        Analyzes the dataframe
    """

    lab_data_dataframe = pd.read_csv(FILENAME)   
     
    # 1. Print only the salary columns
    print(lab_data_dataframe[['salary']]) 
    
    # 2. print only the education column
    print(lab_data_dataframe[['education']]) 
        
    # 3. Print the rows that have their salary value above or equal 127
    salary_greater_than_ = lab_data_dataframe[lab_data_dataframe['salary']>=int(127)]
    print(salary_greater_than_)

    average_salary =  lab_data_dataframe['salary'].mean()
    
   # 4. Print the rows that have their salary value above the average salary
    salary_greater_than_mean = lab_data_dataframe[lab_data_dataframe['salary'] > average_salary]
    print(salary_greater_than_mean)
    

dataframe_analyzer(FILENAME=FILENAME)



