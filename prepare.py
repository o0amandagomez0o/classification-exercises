import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import acquire

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import host, user, password



def clean_iris(df):
    """
    clean_iris will take an acquired df and 
    remove `species_id` and `measurement_id` columns and 
    rename `species_name` column to just `species` and
    encode 'species_name' column into TWO new columns
    
    return: single cleaned dataframe
    """
    
    dropcols = ['species_id', 'measurement_id']
    df.drop(columns= dropcols, inplace=True)
    df.rename(columns={'species_name': 'species'}, inplace=True)
    dummy_sp = pd.get_dummies(df[['species']], drop_first=True)
    return pd.concat([df, dummy_sp], axis =1)


def prep_iris(df):
    """
    prep_iris will take one argument(df) and 
    run clean_iris to remove/rename/encode columns
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """
    iris_df = clean_iris(df)
    train_validate, test = train_test_split(iris_df, test_size=0.2, random_state=3210, stratify=iris_df.species)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210, stratify=train_validate.species)
    return train, validate, test