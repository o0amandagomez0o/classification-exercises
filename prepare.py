import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import acquire

from sklearn.preprocessing import LabelEncoder
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


def handle_missing_values(df):
    return df.assign(
        embark_town=df.embark_town.fillna('Other'),
        embarked=df.embarked.fillna('O'),
    )

def remove_columns(df):
    return df.drop(columns=['deck'])

def encode_embarked(df):
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(embarked_encode = encoder.transform(df.embarked))

def prep_titanic_data(df):
    df = df\
        .pipe(handle_missing_values)\
        .pipe(remove_columns)\
        .pipe(encode_embarked)
    return df

def train_validate_test_split(df, seed=3210):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.survived
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.survived,
    )
    return train, validate, test



def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name for the stratify_by argument
    """

    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])

    return train, validate, test



def clean_titanic(df):
    """
    clean_titanic will take an acquired df and 
    covert sex column to boolean 'is_female' col
    encode "embarked" & "class" columns & add them to the end
    and drop 'age' & 'deck' cols due to NaNs
    'passenger_id' due to un-necessity
    'embark_town', 'embarked', 'sex', 'pclass', 'class' due to redundancy/enew encoded cols
    
    return: single cleaned dataframe
    """
    
    
    df["is_female"] = df.sex == "Female"
    embarked_dummies = pd.get_dummies(df.embarked, prefix='Embarked', drop_first=True)
    class_dummies = pd.get_dummies(df.pclass, prefix='class', drop_first=True)

    dropcols = ['deck', 'age', 'embark_town', 'passenger_id', 'embarked', 'sex', 'pclass', 'class']
    df.drop(columns= dropcols, inplace=True)

    return pd.concat([df, embarked_dummies, class_dummies], axis =1)




def prep_titanic(df):
    """
    prep_iris will take one argument(df) and 
    run clean_iris to remove/rename/encode columns
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """
    df = clean_titanic(df)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210, stratify=df.survived)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210, stratify=train_validate.survived)
    return train, validate, test



def train_validate_test_split(df, target, seed=3210):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test