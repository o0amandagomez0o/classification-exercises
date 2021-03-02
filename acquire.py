#Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame. Obtain your data from the Codeup Data Science Database.
import pandas as pd
import numpy as np
import os
from env import host, user, password




def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
#def get_titanic_data():
#    '''
#    This function reads in the titanic data from the Codeup db
#    and returns a pandas DataFrame with all columns.
#    '''
#    sql_query = 'SELECT * FROM passengers'
#    return pd.read_sql(sql_query, get_connection('titanic_db'))



def get_iris_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = 'select * from species join measurements using (species_id);'
    return pd.read_sql(sql_query, get_connection('iris_db'))



def rf_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in titanic df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('titanic.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = get_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('titanic.csv', index_col=0)
        
    return df



def rf_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in iris df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('iris.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = get_iris_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('iris.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('iris.csv', index_col=0)
        
    return df




def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))










