# 1. IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Make the plots look nice
sns.set_theme(style="whitegrid")

# 2. LOAD AND PREPARE THE DATA

# Load the built-in wine dataset
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# We keep the actual labels (0, 1, or 2) just to color our graph later 
# and prove that our unsupervised math worked!
df['Actual_Wine_Class'] = wine_data.target 

print(f"Dataset Shape: {df.shape} (178 wines, 13 chemical features)")

# 3. SCALE THE DATA (CRUCIAL STEP!)

# Alcohol is measured in percentages (e.g., 13.2), but Magnesium is in mg (e.g., 100).
# If we don't scale them, the algorithm will think Magnesium is 100x more important!
# StandardScaler makes all features have a mean of 0 and a standard deviation of  the same scale.
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('Actual_Wine_Class', axis=1)), 
                         columns=wine_data.feature_names)

# ==========================================
# 4. APPLY PCA (Principal Component Analysis)
# ==========================================
# We tell PCA to squash the 13 features down to 2 "Principal Components"
pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(df_scaled)

# Put the results in a dataframe for easy plotting
pca_df = pd.DataFrame(pca_results, columns=['PCA_Component_1', 'PCA_Component_2'])
pca_df['Actual_Wine_Class'] = df['Actual_Wine_Class']

# ==========================================
# 5. APPLY t-SNE
# ==========================================
# 'perplexity' is like the "number of close neighbors" the algorithm considers. 
# 30 is a great default for datasets of this size.
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
tsne_results = tsne.fit_transform(df_scaled)

# Put the results in a dataframe for easy plotting
tsne_df = pd.DataFrame(tsne_results, columns=['tSNE_Component_1', 'tSNE_Component_2'])
tsne_df['Actual_Wine_Class'] = df['Actual_Wine_Class']

# ==========================================
# 6. VISUALIZE AND COMPARE THE RESULTS
# ==========================================
# Create a figure with 2 plots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: PCA ---
sns.scatterplot(
    x='PCA_Component_1', 
    y='PCA_Component_2', 
    hue='Actual_Wine_Class', 
    palette='viridis', 
    data=pca_df, 
    s=100, # size of the dots
    ax=axes[0]
)
axes[0].set_title('PCA: Squashing data by "Variance"\n(Notice some overlap between classes)', fontsize=14)
axes[0].set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% of variance)')
axes[0].set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% of variance)')

# --- Plot 2: t-SNE ---
sns.scatterplot(
    x='tSNE_Component_1', 
    y='tSNE_Component_2', 
    hue='Actual_Wine_Class', 
    palette='viridis', 
    data=tsne_df, 
    s=100, 
    ax=axes[1]
)
axes[1].set_title('t-SNE: Grouping data by "Similarity"\n(Notice the tight, distinct clusters!)', fontsize=14)
axes[1].set_xlabel('t-SNE Component 1')
axes[1].set_ylabel('t-SNE Component 2')

plt.suptitle("PCA vs t-SNE: Can we separate the wines?", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

print("Project Complete! Look at the graphs above.")
