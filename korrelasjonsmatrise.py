import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Hent inn datasettet
filsti = 'rensede_lopeokter_4.csv'
df = pd.read_csv(filsti)

# 2. Velg ut de relevante numeriske variablene for analysen
valgte_variabler = [
    'Distanse',
    'Bevegelsestid',
    'Gjennomsnittlig fart',
    'Stigning',
    'Gjennomsnittlig puls',
    'Kalorier'
]

# Kortere norske navn til aksene i figuren
nye_navn = {
    'Distanse': 'Distanse',
    'Bevegelsestid': 'Tid',
    'Gjennomsnittlig fart': 'Fart',
    'Stigning': 'Stigning',
    'Gjennomsnittlig puls': 'Puls',
    'Kalorier': 'Kalorier'
}

df_korrelasjon = df[valgte_variabler].rename(columns=nye_navn)

# 3. Beregn Pearsons korrelasjonsmatrise
korrelasjonsmatrise = df_korrelasjon.corr(method='pearson')


# 4. Sett opp figuren
plt.figure(figsize=(8, 6))
sns.set_theme(style="white")

# 5. Tegn heatmappet med Seaborn
figur = sns.heatmap(
    korrelasjonsmatrise,
    annot=etiketter_norsk, # Bruker tabellen med norske desimalkomma
    fmt="",                # Tom streng siden vi allerede har formatert tallene
    cmap='Blues',          # Ren blå fargeskala
    vmin=0, vmax=1,        # Skala fra 0 til 1
    square=True,
    linewidths=.5,
    cbar_kws={"label": "Pearsons korrelasjonskoeffisient (r)"}
)

plt.title("Korrelasjonsmatrise for treningsvariabler ($N=178$)", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 6. Lagre figuren høyoppløselig
plt.tight_layout()
plt.savefig('heatmap_korrelasjon.png', dpi=300, bbox_inches='tight')
plt.show()