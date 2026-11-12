# ISTG1003-Prosjekt/Logisk regresjon av treningssett

"""
Påkrevde pakker for å gjøre klasifikasjon av datasettet.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"""
Funksjon for å hente inn data fra CSV-filen, med feilhånderting for 
feil som kan oppstå under innlasting.
"""

def extract_lop_data(file_path, collums):
    """
    Extracts LOP data from a CSV file and returns it as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file containing LOP data.

    Returns:
    pd.DataFrame: A DataFrame containing the extracted LOP data.
    """
    try:
        lop_data = pd.read_csv(file_path, usecols=collums)
        
        return lop_data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    
    except Exception as e:
        print(f"An error occurred while extracting LOP data: {e}")
        return None
# Definerer kolonner som skal brukes i klassifiseringen.

data_collums = [
    "Distanse",
    "Bevegelsestid",
    "Gjennomsnittlig puls",
    "Stigning",
    "Kalorier"
]

# Filstien for datasettet

filepath = "ISTG1003-Prosjekt\\data\\rensede_lopeokter_4.csv"

# Kaller på funksjon for å hente inn data fra CSV-filen, og legger med de spesifike
# kolonnene som skal brukes.

df = extract_lop_data(filepath, data_collums)

# Gir kolonnene med kopplekse navn mer lesbare navn for videre bruk.

df_cleaned = df.rename(columns={
    "Distanse": "Distanse",
    "Bevegelsestid": "Tid",
    "Gjennomsnittlig puls": "Puls",
    "Stigning": "Stigning",
    "Kalorier": "Kalorier"
})

"""
Sjekker om df er none før videre behandling av dataen.
"""

if df is not None:
    df_cleaned = df.rename(columns={
        "Bevegelsestid": "Tid",
        "Gjennomsnittlig puls": "Puls"
    })

   
    """
    DATASPLITTING (60% Trening, 20% Validering, 20% Test)
    """

    df_train_val, df_test = train_test_split(df_cleaned, test_size=0.20, random_state=42)
   
    df_train, df_val = train_test_split(df_train_val, test_size=0.25, random_state=42)

    # Printer antall økter i hvert datasett for å bekrefte splittingen.
    print(f"Antall økter: Trening={len(df_train)}, Validering={len(df_val)}, Test={len(df_test)}")


    # Visualiserer treningsdataene med et pairplot for å se på sammenhenger mellom variablene.
    sns.pairplot(df_train)
    plt.suptitle("Pairplot av treningsdata", y=1.02)
    plt.show()

    #Tilpass en lineær regresjonsmodell ved hjelp av biblioteket statsmodels
    formel = 'Kalorier ~ Distanse + Tid + Stigning'
    model = smf.ols(formula=formel, data=df_train).fit()

   #Skriv ut regresjonsresultatene og evaluer modellen på valideringssettet og testsett.
    print("\n--- Regresjonsresultater (fra treningssettet) ---")
    print(model.summary())


    # Evaluer modellen på valideringssettet og testsettet

    val_pred = model.predict(df_val)
    val_rmse = np.sqrt(mean_squared_error(df_val['Kalorier'], val_pred))
    val_r2 = r2_score(df_val['Kalorier'], val_pred)
    print(f"\n[Validering] RMSE: {val_rmse:.2f} kcal | R^2: {val_r2:.3f}")

   # Evaluer modellen på testsettet

    test_pred = model.predict(df_test)
    test_rmse = np.sqrt(mean_squared_error(df_test['Kalorier'], test_pred))
    test_r2 = r2_score(df_test['Kalorier'], test_pred)
    print(f"[Testsett]   RMSE: {test_rmse:.2f} kcal | R^2: {test_r2:.3f}")


    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    """
    Plotting av resultatene fra regresjonsmodellen
    """

    ax[0].scatter(model.fittedvalues, model.resid, alpha=0.7)
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].set_title('Tukey-Anscombe plott')
    ax[0].set_xlabel('Predikerte verdier')
    ax[0].set_ylabel('Residualer')
    ax[0].grid(True)

    sm.qqplot(model.resid, line='45', fit=True, ax=ax[1])
    ax[1].set_title('QQ-plott av residualene')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()