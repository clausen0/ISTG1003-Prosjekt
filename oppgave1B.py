# ISTG1003-Prosjekt/Kvalifikasjon av treningssett

"""
Påkrevde pakker for å gjøre klasifikasjon av datasettet.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

"""
Funksjon for å hente inn data fra CSV-filen, med feilhånderting for 
feil som kan oppstå under innlasting.
"""

def extract_lop_data(file_path, columns):
    try:
        lop_data = pd.read_csv(file_path, usecols=columns)
        return lop_data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while extracting LOP data: {e}")
        return None


# Definerer kolonner som skal brukes i klassifiseringen.

data_columns = [
    "Distanse",
    "Bevegelsestid",
    "Gjennomsnittlig puls",
    "Stigning"
]


# Filstien for datasettet


filepath = "ISTG1003-Prosjekt\\data\\rensede_lopeokter_4.csv"


# Funksjon for å hente inn data fra CSV-filen, og legger med de spesifike
# kolonnene som skal brukes.

df = extract_lop_data(filepath, data_columns)

"""
Sjekker om df er none før videre behandling av dataen.
"""

if df is not None:

    # Gir kolonnene med kopplekse navn mer lesbare navn for videre bruk.

    df_cleaned = df.rename(
        columns={
            "Bevegelsestid": "Tid", 
             "Gjennomsnittlig puls": "Puls", 
             "Gjennomsnittlig fart": "Fart"
            }
    )

    # Beregn Fart i km/t
    df_cleaned["Fart"] = df_cleaned["Distanse"] / (df_cleaned["Tid"] / 3600)

    # Opprett fasitkolonnen 'Intensitet'
    df_cleaned["Intensitet"] = (df_cleaned["Puls"] > 158.5).astype(int)

    """
    DATASPLITTING (60% Trening, 20% Validering, 20% Test)
    """
    df_train_val, df_test = train_test_split(
        df_cleaned,
        test_size=0.20,
        random_state=42,
        stratify=df_cleaned["Intensitet"],
    )
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.25,
        random_state=42,
        stratify=df_train_val["Intensitet"],
    )

    features = ["Distanse", "Tid", "Fart", "Stigning"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    X_val = scaler.transform(df_val[features])
    X_test = scaler.transform(df_test[features])

    y_train = df_train["Intensitet"]
    y_val = df_val["Intensitet"]
    y_test = df_test["Intensitet"]

    """
    Klassifisering med k-NN 
    """
    knaboer = np.arange(1, 20, step=2)
    val_feilrate = np.empty(len(knaboer))

    for i, k in enumerate(knaboer):
        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        knn.fit(X_train, y_train)
        val_feilrate[i] = 1 - knn.score(X_val, y_val)

    optimal_k = knaboer[np.argmin(val_feilrate)]
    print(f"Optimal k valgt ut fra valideringssettet: k = {optimal_k}")
    print(f"Laveste feilrate på valideringssettet (k-NN): {np.min(val_feilrate):.4f}")

    knn_best = KNeighborsClassifier(n_neighbors=optimal_k, p=2)
    knn_best.fit(X_train, y_train)
    y_pred_knn = knn_best.predict(X_test)

    """
    Klasifisering med Logistisk Regresjon
    """
    formel = "Intensitet ~ Distanse + Tid + Fart + Stigning"
    logit_mod = smf.logit(formel, data=df_train).fit()

    print("\n--- Logistisk Regresjon Resultater ---")
    print(logit_mod.summary())

    # Skriver ut odds-faktorer (exp(beta)) som gjennomgått i forelesning
    print("\n--- Odds-faktorer (exp(beta)) ---")
    print(np.exp(logit_mod.params))

    cutoff = 0.5
    val_pred_prob = logit_mod.predict(exog=df_val)
    y_val_pred_logit = np.where(val_pred_prob > cutoff, 1, 0)
    val_feilrate_logit = 1 - accuracy_score(y_true=y_val, y_pred=y_val_pred_logit)
    print(f"\nFeilrate på valideringssettet (Logistisk regresjon): {val_feilrate_logit:.4f}")

    test_pred_prob = logit_mod.predict(exog=df_test)
    y_pred_logit = np.where(test_pred_prob > cutoff, 1, 0)

    """
    Evaluering av begge modellene på testsettet
    """
    feilrate_knn = 1 - accuracy_score(y_true=y_test, y_pred=y_pred_knn)
    feilrate_logit = 1 - accuracy_score(y_true=y_test, y_pred=y_pred_logit)

    print("\n--- EVALUERING PÅ TESTDATA ---")
    print(
        f"k-NN (k={optimal_k}) -> Feilrate: {feilrate_knn:.4f} | "
        f"Nøyaktighet: {accuracy_score(y_test, y_pred_knn):.2%}"
    )
    print(
        f"Logistisk Regresjon -> Feilrate: {feilrate_logit:.4f} | "
        f"Nøyaktighet: {accuracy_score(y_test, y_pred_logit):.2%}"
    )

    cm = confusion_matrix(y_test, y_pred_logit)

    """
    Plotting av forvirringsmatrise fra klassifikasjonen
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rolig/Moderat (0)", "Hard (1)"],
        yticklabels=["Rolig/Moderat (0)", "Hard (1)"],
    )

    plt.xlabel("Predikert klasse")
    plt.ylabel("Sann klasse")
    plt.title("Forvirringsmatrise (Testsett - Logistisk Regresjon)")
    plt.show()