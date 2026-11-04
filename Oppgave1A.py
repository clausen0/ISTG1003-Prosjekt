import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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

filepath = "ISTG1003-Prosjekt\\data\\rensede_lopeokter_4.csv"

df = extract_lop_data(filepath, data_collums)

df_cleaned = df.rename(columns={
    "Distanse": "Distanse",
    "Bevegelsestid": "Tid",
    "Gjennomsnittlig puls": "Puls",
    "Stigning": "Stigning",
    "Kalorier": "Kalorier"
})

if df is not None:
    df_cleaned = df.rename(columns={
        "Bevegelsestid": "Tid",
        "Gjennomsnittlig puls": "Puls"
    })

    ## Split the data into training, validation, and test sets
    df_train_val, df_test = train_test_split(df_cleaned, test_size=0.20, random_state=42)
   
    df_train, df_val = train_test_split(df_train_val, test_size=0.25, random_state=42)

    ## Print the number of sessions in each set
    print(f"Antall økter: Trening={len(df_train)}, Validering={len(df_val)}, Test={len(df_test)}")


    ## Visualize the training data using a pairplot
    sns.pairplot(df_train)
    plt.suptitle("Pairplot av treningsdata", y=1.02)
    plt.show()

    ## Fit a linear regression model using statsmodels
    formel = 'Kalorier ~ Distanse + Tid + Stigning'
    model = smf.ols(formula=formel, data=df_train).fit()

   ## Print the regression results and evaluate the model on validation and test sets
    print("\n--- Regresjonsresultater (fra treningssettet) ---")
    print(model.summary())


    ## Evaluate the model on validation and test sets
    val_pred = model.predict(df_val)
    val_rmse = np.sqrt(mean_squared_error(df_val['Kalorier'], val_pred))
    val_r2 = r2_score(df_val['Kalorier'], val_pred)
    print(f"\n[Validering] RMSE: {val_rmse:.2f} kcal | R^2: {val_r2:.3f}")

   ## Evaluate the model on the test set
    test_pred = model.predict(df_test)
    test_rmse = np.sqrt(mean_squared_error(df_test['Kalorier'], test_pred))
    test_r2 = r2_score(df_test['Kalorier'], test_pred)
    print(f"[Testsett]   RMSE: {test_rmse:.2f} kcal | R^2: {test_r2:.3f}")


    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ## Plot the residuals
    ax[0].scatter(model.fittedvalues, model.resid, alpha=0.7)
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].set_title('Tukey-Anscombe plott')
    ax[0].set_xlabel('Predikerte verdier')
    ax[0].set_ylabel('Residualer')
    ax[0].grid(True)

   ### Plot the QQ plot of the residuals
    sm.qqplot(model.resid, line='45', fit=True, ax=ax[1])
    ax[1].set_title('QQ-plott av residualene')
    ax[1].grid(True)

    ## Plot the predicted vs actual values for the test set
    plt.tight_layout()
    plt.show()