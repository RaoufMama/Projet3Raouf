import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_excel('ecommerce_transactions.xlsx')

# 1. Aperçu des données
print(df.head())

# 2. Dimensions et types des colonnes
print(df.shape)
print(df.dtypes)

# 3. Identifier les colonnes avec des valeurs manquantes
print(df.isna().sum())

# Stratégie de gestion des NaN (exemple : remplir avec la médiane pour les montants)
df.fillna({'Montant total (€)': df['Montant total (€)'].median()}, inplace=True)

# 4. Suppression des doublons
df.drop_duplicates(inplace=True)

# 5. Création de la colonne ‘Année-Mois’
df['Date'] = pd.to_datetime(df['Date'])
df['Année-Mois'] = df['Date'].dt.to_period('M')

# 6. Top 5 des pays générant le plus de chiffre d’affaires
print(df.groupby('Pays')['Montant total (€)'].sum().nlargest(5))

# 7. Chiffre d’affaires total par catégorie de produit
print(df.groupby('Catégorie')['Montant total (€)'].sum())

# 8. Marques les plus vendues par catégorie
print(df.groupby(['Catégorie', 'Marque'])['Quantité'].sum().reset_index().sort_values(['Catégorie', 'Quantité'], ascending=[True, False]))

# 9. Méthodes de paiement les plus utilisées par pays
print(df.groupby(['Pays', 'Méthode de paiement']).size().unstack().fillna(0))

# 10. Dépense moyenne par client et top 10 clients
df_client = df.groupby('Client ID')['Montant total (€)'].sum()
print(df_client.mean())
print(df_client.nlargest(10))

# 11. Note moyenne par catégorie et par pays
print(df.groupby(['Catégorie', 'Pays'])['Note client'].mean().unstack())

# 12. Analyse des notes manquantes
print(df[df['Note client'].isna()].groupby(['Catégorie', 'Pays']).size())

# 13. Statistiques sur les montants totaux
print(f"Moyenne: {np.mean(df['Montant total (€)'])}, Médiane: {np.median(df['Montant total (€)'])}, Std: {np.std(df['Montant total (€)'])}")

# 14. Définition des clients fidèles
df['Client fidèle'] = df.groupby('Client ID')['Client ID'].transform('count') > 5

# 15. Graphique CA mensuel
sns.barplot(x=df['Année-Mois'].astype(str), y=df['Montant total (€)'], estimator=sum)
plt.xticks(rotation=90)
plt.show()

# 16. Répartition des ventes par catégorie
plt.pie(df.groupby('Catégorie')['Montant total (€)'].sum(), labels=df['Catégorie'].unique(), autopct='%1.1f%%')
plt.show()

# 17. Boxplot montant des commandes par méthode de paiement
sns.boxplot(x='Méthode de paiement', y='Montant total (€)', data=df)
plt.xticks(rotation=90)
plt.show()

# 18. Heatmap des notes moyennes
sns.heatmap(df.groupby(['Pays', 'Catégorie'])['Note client'].mean().unstack(), cmap='coolwarm', annot=True)
plt.show()

# 19. Scatter plot quantité vs montant total
sns.scatterplot(x='Quantité', y='Montant total (€)', data=df)
plt.show()

# Sauvegarde des données nettoyées
df.to_excel('ecommerce_transactions_clean.xlsx', index=False)
