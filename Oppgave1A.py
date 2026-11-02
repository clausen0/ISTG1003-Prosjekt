import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


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
    "Gjennomsnittlig puls",
    "Stigning",
    "Kalorier"
]

filepath = "ISTG1003-Prosjekt\\data\\rensede_lopeokter_3.csv"

df_cleaned = extract_lop_data(filepath, data_collums)

df_cleaned = df_cleaned.rename(columns={
    "Distanse": "Distanse",
    "Bevegelsestid": "Tid",
    "Gjennomsnittlig puls": "Puls",
    "Stigning": "Stigning",
    "Kalorier": "Kalorier"
})

if df_cleaned is not None:
    distanse = df_cleaned["Distanse"]
    tid = df_cleaned["Tid"]
    puls = df_cleaned["Puls"]
    stigning = df_cleaned["Stigning"]
    kalorier = df_cleaned["Kalorier"]

    print("Data extracted successfully:")
    print(df_cleaned.head())  


print("---Oppsumerging av data---")
print(df_cleaned.describe())

sns.pairplot(df_cleaned)
plt.suptitle("Pairplot of LOP Data", y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='Distanse', y='Kalorier')
plt.title('Sammenheng mellom puls og kalorier')
plt.xlabel('Gjennomsnittlig puls (bpm)')
plt.ylabel('Kalorier (kcal)')
plt.grid(True)
plt.show()

Regresjon_formel = 'Kalorier ~ Distanse + Tid + Puls + Stigning'

model = smf.ols(formula=Regresjon_formel, data=df_cleaned).fit()
resultat = model.summary()

print(resultat)