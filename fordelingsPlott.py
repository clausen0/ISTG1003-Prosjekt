import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Les inn datasettet
df = pd.read_csv('rensede_lopeokter.csv')

# 2. Sett stil og opprett to plott ved siden av hverandre
sns.set_theme(style='whitegrid')
fig, (ax_distanse, ax_puls) = plt.subplots(1, 2, figsize=(11, 4.5))

# 3. Panel A: Fordeling av distanse
sns.histplot(
    df['Distanse'],
    bins=18,
    kde=True,
    color='#2b5c8f',
    ax=ax_distanse
)
ax_distanse.set_title(
    'A: Fordeling av Distanse (Høgreskjev)',
    fontsize=11,
    fontweight='bold'
)
ax_distanse.set_xlabel('Distanse (km)')
ax_distanse.set_ylabel('Antall økter')

# 4. Panel B: Fordeling av puls med intensitetsgrense
sns.histplot(
    df['Gjennomsnittlig puls'],
    bins=18,
    kde=True,
    color='#d95f02',
    ax=ax_puls
)
ax_puls.axvline(
    158.5,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label='Intensitetsgrense (158.5 bpm)'
)
ax_puls.set_title(
    'B: Fordeling av Puls (Tilnærmet symmetrisk)',
    fontsize=11,
    fontweight='bold'
)
ax_puls.set_xlabel('Gjennomsnittlig puls (bpm)')
ax_puls.set_ylabel('Antall økter')
ax_puls.legend(loc='upper right')

# 5. Lagre og vis figuren
plt.tight_layout()
plt.savefig('histogrammer_fordeling.png', dpi=300)
plt.show()