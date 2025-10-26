import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
%pip install scikit-learn
from sklearn.impute import SimpleImputer
%pip install scipy
from scipy import stats
%pip install ipywidgets
import ipywidgets as widgets
from ipywidgets import interact
%pip install plotly
import plotly.express as px

# Importer les données
df = pd.read_csv('/Users/maximebarbier/Documents/00 - Repositories/taux-de-pression-et-emploi/data/external/fr-en-taux-de-pression-et-demploi.csv',sep=';')
df.sample(5)

# 1 Nettoryage des données

## 1.1 Vérification des valeurs manquantes

df.isna().mean().sort_values(ascending=False)

### Remplacement des valeurs manquantes par la moyenne de la colonne
imputer = SimpleImputer(strategy='mean')    
df[['Taux de pression']] = imputer.fit_transform(df[['Taux de pression']])

## 1.2 Vérification des doublons
df.duplicated().value_counts(normalize=True)

### Pas de doublons détectés

## 1.3 Vérification des types de données
df.dtypes

### Les types de données semblent corrects

## 1.4 Vérification des valeurs aberrantes
numeric_col = ['Taux de pression', "Taux d'emploi"]

# Calculer le nombre de lignes et colonnes
n_cols = 3
n_rows = (len(numeric_col) + n_cols - 1) // n_cols  # 3 graphiques par ligne

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

# Aplatir correctement les axes (même si 1 seule ligne)
axes = np.array(axes).reshape(-1)

for i, col in enumerate(numeric_col):
    ax = axes[i]
    # Boxplot avec détection des outliers
    ax.boxplot(df[col].dropna(),
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))

    ax.set_title(f"{col.replace('_', ' ').title()}", fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Ajouter le nombre d'observations
    n_obs = df[col].notna().sum()
    ax.text(0.02, 0.98, f'N = {n_obs}',
            transform=ax.transAxes, verticalalignment='top')

# Masquer les axes inutilisés
for i in range(len(numeric_col), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Calculer le Z-scores :
df_numeric = df[numeric_col]

for col in numeric_col:
  z_scores = np.abs(stats.zscore(df_numeric[col]))
  outliers = df[z_scores > 3]
  nb_outliers = len(outliers)
  nb_total = len(df)
  pourcentage = (nb_outliers / nb_total) * 100

  print(col)
  print('='*18)
  print(f"Nombre d'outliers : {nb_outliers}")
  print(f"Nombre total d'observations : {nb_total}")
  print(f"Pourcentage d'outliers : {pourcentage:.2f}%")
  print("")

### Calcul du Z-score pour détecter les valeurs aberrantes pour le taux de pression
df['z_score'] = (df['Taux de pression'] - df['Taux de pression'].mean()) / df['Taux de pression'].std()
outliers = df[np.abs(df['z_score']) > 3]
outliers
### Les valeurs aberrantes détectées seront conservées pour l'analyse
df.drop(columns=['z_score'], inplace=True)

# 2 Analyse exploratoire des données (EDA)
## 2.1 Statistiques descriptives
df.describe()

df['Type de diplôme'].value_counts(normalize=True)
df['Filière'].value_counts(normalize=True)

df.pivot_table(index='Filière', values=['Taux de pression', "Taux d'emploi"], aggfunc=['mean', 'median', 'std'])

## 2.2 Visualisations
### Distribution des variables numériques
plt.figure(figsize=(12, 5))
for i, col in enumerate(numeric_col):
    plt.subplot(1, 2, i + 1)
    sns.histplot(df[col], kde=True, color='skyblue', bins=30)
    plt.title(f'Distribution de {col.replace("_", " ").title()}', fontweight='bold')
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Fréquence')
plt.tight_layout()
plt.show()

### Relation entre le taux de pression et le taux d'emploi
# Liste unique des filières
filieres = sorted(df['Filière'].dropna().unique())

# Widget : liste à cocher multiple
filiere_selector = widgets.SelectMultiple(
    options=filieres,
    value=filieres,  # toutes sélectionnées par défaut
    description='Filières',
    layout={'width': '300px', 'height': '200px'}
)

# Fonction d'affichage dynamique
def update_plot(selected_filieres):
    plt.figure(figsize=(8, 6))
    data = df[df['Filière'].isin(selected_filieres)]
    
    sns.scatterplot(
        data=data,
        x='Taux de pression',
        y="Taux d'emploi",
        hue='Filière',
        palette='Set2',
        alpha=0.7
    )

    plt.title('Relation entre le Taux de Pression et le Taux d\'Emploi', fontweight='bold')
    plt.xlabel('Taux de Pression')
    plt.ylabel('Taux d\'Emploi')
    plt.legend(title='Filière', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Liaison du widget à la fonction
interact(update_plot, selected_filieres=filiere_selector)
