import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def extract_lop_data(file_path, collums, papername=0):
    """
    Extracts LOP data from a CSV file and returns it as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file containing LOP data.

    Returns:
    pd.DataFrame: A DataFrame containing the extracted LOP data.
    """
    try:
        # Read the CSV file into a DataFrame
        lop_data = pd.read_csv(file_path, usecols=collums)
        
        # Perform any necessary data cleaning or processing here
        # For example, you might want to drop missing values or filter specific columns
        
        return lop_data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    
    except Exception as e:
        print(f"An error occurred while extracting LOP data: {e}")
        return None

data_collums = [
    "Distanse",
    "Bevegelsestid",
    "Gjennomsnittlig fart",
    "Gjennomsnittlig puls",
    "Kalorier"
]

filepath = "ISTG1003-Prosjekt\\data\\rensede_lopeokter_3.csv"

df = extract_lop_data(filepath, data_collums)

if df is not None:
    distanse = df["Distanse"]
    tid = df["Bevegelsestid"]
    fart = df["Gjennomsnittlig fart"]
    puls = df["Gjennomsnittlig puls"]
    kalorier = df["Kalorier"]

    print("Data extracted successfully:")
    print(df.head())  