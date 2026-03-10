import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import itertools
from scipy.stats import zscore
from pyampute.exploration.mcar_statistical_tests import MCARTest

df = pd.read_csv("/home/oscar/Documents/AM/Proyecto/customer segmentation (no supervisado)/Test.csv")

numeric_features = df.select_dtypes(include=['number']).drop(columns=['ID']).columns
categoric_features = df.select_dtypes(include=['object', 'category']).columns

# Descripción del Dataset
include = ['object', 'float', 'int', 'bool', 'string']
resume = df.describe(include=include)
print(resume, end = "\n\n")

df.info()

# Distribución categórica de las variables
for col in categoric_features:
    print(df[col].value_counts(), end="\n\n")

# Visualización de las variables numéricas  
fig, ax = plt.subplots(figsize=(4,4))
df.drop('ID', axis=1).hist(ax=ax)
plt.tight_layout()
plt.show()

# Visualización de las variables categóricas
df_melted = df[categoric_features].melt(var_name='variable', value_name='category')
sns.catplot(data=df_melted, x='category', col='variable', col_wrap=3, 
            kind='count', sharex=False, height=4, aspect=1.5)
plt.tight_layout()
plt.show()

# Detección de outliers con z-score
for j in numeric_features:
    z_scores_numeric = zscore(df[j])
    abs_z_scores = np.abs(z_scores_numeric)
    
    outliers = df[abs_z_scores > 3]
    outliers.head()
    print(f"Número de outliers en {j}: {len(outliers)}")

# Mapeo de valores faltantes
plt.figure(figsize=(6,10))
sns.heatmap(df.isnull(), cmap='viridis', cbar =False, yticklabels=False)
plt.title("Datos faltantes con un Heatmap")
plt.show()

# Test de detección de MCAR
df_int = df.drop(columns=['ID']).select_dtypes(include=['number'])

mt = MCARTest(method="little")
p_value = mt.little_mcar_test(df_int)

print("El valor de p es:", p_value)

alpha = 0.05
if p_value > alpha:
    print("Los datos faltantes podrían ser MCAR")
else:
    print("Los datos faltantes no son MCAR")

# Correlación de Pearson entre las variables numéricas
corr_matrix = df[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title('Correlación entre variables numéricas')
plt.show()

# ANOVA para probar diferencias entre grupos
groups =[df[df['Spending_Score']==cat]['Age'].dropna() for cat in df['Spending_Score'].unique()]
f_stat, p_val = stats.f_oneway(*groups)
print(f'ANOVA - p-valor: {p_val}')

# Asociación entre variables categóricas
unique_counts = {col: df[col].nunique() for col in categoric_features}

for col1, col2 in itertools.combinations(categoric_features, 2):
    # Solo si ambas tienen el mismo número de categorías 
    if unique_counts[col1] == unique_counts[col2]:
        crosstab = pd.crosstab(df[col1], df[col2])
        
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Frecuencia de {col1} vs {col2}")
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.tight_layout()
        plt.show()

chi2, p, dof, expected = stats.chi2_contingency(crosstab)
print(f'Chi-Cuadrado p-valor: {p}')

# Mapeo ordinal 
mapa_gasto = {'Low': 1, 'Medium': 2, 'High': 3}
df['Gasto_ordinal'] = df['Spending_Score'].map(mapa_gasto)

# Correlación de Spearman con las numéricas
corr_spearman = df[['Age', 'Work_Experience', 'Family_Size', 'Gasto_ordinal']].corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, cmap='Blues')
plt.title('Correlación de Spearman con un mapeo ordinal')
plt.show()

# Pairplot con hue
sns.pairplot(df, vars=numeric_features, hue='Spending_Score', diag_kind='kde', palette='Set2')
plt.suptitle('Relaciones entre variables numéricas por Spending Score', y=1.02)
plt.show()

