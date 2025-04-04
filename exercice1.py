import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le fichier Excel
df = pd.read_excel('employes_dataset.xlsx')

# 1. Afficher les 10 premières lignes
print(df.head(10))

# 2. Afficher les noms de colonnes
print(df.columns)

# 3. Compter le nombre d’hommes et de femmes
print(df['Sexe'].value_counts())

# 4. Identifier les 5 pays les plus représentés
print(df['Pays'].value_counts().head(5))

# 5. Statistiques sur les salaires
salaires = df['Salaire (€)']
print(f"Moyenne: {np.mean(salaires)}, Médiane: {np.median(salaires)}, Min: {np.min(salaires)}, Max: {np.max(salaires)}, Écart-type: {np.std(salaires)}")

# 6. Âge moyen par département
print(df.groupby('Département')['Âge'].mean())

# 7. Ville avec le plus grand nombre d’employés
print(df['Ville'].value_counts().idxmax())

# 8. Top 10 employés les mieux payés
print(df.nlargest(10, 'Salaire (€)'))

# 9. Nombre d’employés par département et sexe
print(df.pivot_table(index='Département', columns='Sexe', aggfunc='size', fill_value=0))

# 10. Graphique de distribution des âges
sns.histplot(df['Âge'], bins=20, kde=True)
plt.show()

# 11. Colonnes avec valeurs manquantes
print(df.isna().sum())

# 12-13. Remplacement des NaN en Télétravail (%)
df['Télétravail (%)'].fillna(df['Télétravail (%)'].mean(), inplace=True)

# 14. Suppression des lignes avec 'Performance (Note)' manquante
df.dropna(subset=['Performance (Note)'], inplace=True)

# 15. Stratégie de gestion des valeurs manquantes (exemple: remplissage par médiane)
df['Performance (Note)'].fillna(df['Performance (Note)'].median(), inplace=True)

# 16. Conversion de 'Date d’embauche' en datetime
df["Date d'embauche"] = pd.to_datetime(df["Date d'embauche"])

# 17. Calcul de l'ancienneté
df['Ancienneté (années)'] = (pd.Timestamp.today() - df["Date d'embauche"]).dt.days // 365

# 18. Suppression des doublons
df.drop_duplicates(inplace=True)

# 19. Uniformiser les majuscules
df[['Nom', 'Prénom', 'Ville', 'Pays']] = df[['Nom', 'Prénom', 'Ville', 'Pays']].apply(lambda x: x.str.title())

# 20. Vérification des e-mails valides
df['Email valide'] = df['Email'].str.contains(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=False)

# 21. Suppression des outliers (IQR) en Salaire
Q1 = df['Salaire (€)'].quantile(0.25)
Q3 = df['Salaire (€)'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Salaire (€)'] >= Q1 - 1.5 * IQR) & (df['Salaire (€)'] <= Q3 + 1.5 * IQR)]

# 22. Vérifier si la distribution des âges suit une loi normale
print(np.mean(df['Âge']), np.median(df['Âge']), np.std(df['Âge']))

# 23. Création de la colonne 'prime'
df['Prime'] = np.where((df['Performance (Note)'] >= 4) & (df['Ancienneté (années)'] >= 5), 1000, 0)

# 24. Encodage de 'Sexe'
df['Sexe'] = df['Sexe'].map({'Femme': 0, 'Homme': 1})

# 25. Catégorisation de l’âge
df['Tranche d’âge'] = pd.cut(df['Âge'], bins=[0, 25, 30, 40], labels=['0-25', '26-30', '31-40'])

# 26. Ajout d'une colonne de langages fictifs
df['Langages'] = [['Python', 'Java'], ['C++', 'JavaScript'], ['Python'], ['SQL', 'JavaScript']]
df = df.explode('Langages')

# 27. Transformation avec MultiIndex
df_multi = df.set_index(['Département', 'Sexe'])
print(df_multi.stack().unstack())

# 28. DataFrame des outliers en Salaire
df_outliers = df[(df['Salaire (€)'] < Q1 - 1.5 * IQR) | (df['Salaire (€)'] > Q3 + 1.5 * IQR)]

# 29. Encodage de 'Département'
df = pd.get_dummies(df, columns=['Département'])

# 30. Ajout d’une date fictive et extraction
import datetime
df['Date inscription'] = datetime.datetime.today()
df['Année inscription'] = df['Date inscription'].dt.year
df['Mois inscription'] = df['Date inscription'].dt.month

# 31. Fonction de catégorisation de l’âge
def categoriser_age(age):
    if age < 25:
        return 'Jeune'
    elif age < 40:
        return 'Expérimenté'
    else:
        return 'Senior'
df = df.pipe(lambda d: d.assign(Catégorie_Age=d['Âge'].apply(categoriser_age)))

# 32. Barplot de l’âge moyen par département
plt.figure(figsize=(10, 5))
sns.barplot(
    x=df.groupby('Département')['Âge'].mean().index, 
    y=df.groupby('Département')['Âge'].mean().values
)
plt.xticks(rotation=90)
plt.title("Âge moyen par département")
plt.show()

# 33. Sauvegarder le DataFrame nettoyé
df.to_excel('employes_nettoyé.xlsx', index=False)

# 34. Histogramme de la répartition des âges
plt.figure(figsize=(10, 5))
sns.histplot(df['Âge'], bins=20, kde=True)
plt.title("Répartition des âges")
plt.xlabel("Âge")
plt.ylabel("Nombre d'employés")
plt.show()

# 35. Nombre d’employés par département
plt.figure(figsize=(10, 5))
df['Département'].value_counts().plot(kind='bar', color='teal')
plt.title("Nombre d'employés par département")
plt.xlabel("Département")
plt.ylabel("Nombre d'employés")
plt.xticks(rotation=90)
plt.show()

# 36. Répartition hommes/femmes en camembert
plt.figure(figsize=(6, 6))
df['Sexe'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title("Répartition des employés par sexe")
plt.ylabel("")
plt.show()

# 37. Boxplot des salaires par département
plt.figure(figsize=(12, 6))
sns.boxplot(x='Département', y='Salaire', data=df)
plt.xticks(rotation=90)
plt.title("Distribution des salaires par département")
plt.show()

# 38. Heatmap de corrélation entre colonnes numériques
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Âge', 'Salaire', 'Performance (Note)', 'Télétravail (%)', 'Ancienneté (années)']].corr(), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# 39. Courbe de l’évolution des embauches par année
plt.figure(figsize=(10, 5))
df['Date d’embauche'] = pd.to_datetime(df['Date d’embauche'])
df['Année embauche'] = df['Date d’embauche'].dt.year
df['Année embauche'].value_counts().sort_index().plot(kind='line', marker='o', color='purple')
plt.title("Évolution des embauches par année")
plt.xlabel("Année")
plt.ylabel("Nombre d'embauches")
plt.show()

# 40. Graphique combiné (barres + ligne) : Salaire moyen et performance moyenne par département
fig, ax1 = plt.subplots(figsize=(12, 6))

# Barres pour le salaire moyen
sns.barplot(x=df.groupby('Département')['Salaire'].mean().index, 
            y=df.groupby('Département')['Salaire'].mean().values, 
            color='blue', alpha=0.6, ax=ax1)

# Axe secondaire pour la performance moyenne
ax2 = ax1.twinx()
sns.lineplot(x=df.groupby('Département')['Performance (Note)'].mean().index, 
             y=df.groupby('Département')['Performance (Note)'].mean().values, 
             marker='o', color='red', ax=ax2)

ax1.set_xlabel("Département")
ax1.set_ylabel("Salaire moyen", color='blue')
ax2.set_ylabel("Performance moyenne", color='red')
plt.xticks(rotation=90)
plt.title("Salaire moyen et performance moyenne par département")
plt.show()

# 41. Distribution des salaires avec un KDE plot
plt.figure(figsize=(10, 5))
sns.kdeplot(df['Salaire'], fill=True, color='green')
plt.title("Distribution des salaires")
plt.xlabel("Salaire")
plt.show()

# 42. Carte thermique du nombre d’employés par pays et par sexe
plt.figure(figsize=(12, 6))
heatmap_data = df.pivot_table(index='Pays', columns='Sexe', aggfunc='size', fill_value=0)
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='d')
plt.title("Nombre d’employés par pays et sexe")
plt.show()

# 43. Scatter plot entre âge et salaire, colorié par performance
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Âge'], y=df['Salaire'], hue=df['Performance (Note)'], palette='coolwarm', alpha=0.7)
plt.title("Relation entre âge et salaire selon la performance")
plt.xlabel("Âge")
plt.ylabel("Salaire")
plt.show()
