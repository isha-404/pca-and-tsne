# 🍷 Wine Chemical Profiling: PCA vs. t-SNE

A machine learning project demonstrating dimensionality reduction techniques to visualize and cluster the classic Wine Dataset based on its chemical properties.

## 📌 Project Overview
The goal of this project is to determine if 13 different chemical measurements (such as alcohol, magnesium, and acidity) can naturally separate wines into their distinct cultivar types, *without* using the actual type labels during the grouping process. 

Since humans cannot visualize 13-dimensional data, this project uses two powerful dimensionality reduction techniques to squash the data down to 2D for visualization:
1. **PCA (Principal Component Analysis):** Preserves global variance and the "big picture" structure of the data.
2. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Preserves local neighborhoods, aggressively grouping similar data points together to reveal distinct clusters.

## 📊 The Dataset
This project uses the built-in **Wine Dataset** from `scikit-learn`, eliminating the need for external CSV downloads.
- **Samples:** 178 wines
- **Features:** 13 chemical attributes (Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline)
- **Target:** 3 distinct cultivars of grapes (used *only* for coloring the final plots to validate our unsupervised findings).

## 🛠️ Methodology
1. **Data Loading:** Fetch the dataset directly from `sklearn.datasets`.
2. **Feature Scaling:** Apply `StandardScaler` to normalize all 13 features to a mean of 0 and variance of 1. *This is crucial because features are measured in different units (e.g., percentages vs. milligrams), and unscaled data would bias the algorithms toward larger numbers.*
3. **PCA Transformation:** Reduce the 13 dimensions to 2 Principal Components and visualize the variance explained.
4. **t-SNE Transformation:** Reduce the 13 dimensions to 2 components, experimenting with the `perplexity` hyperparameter to find the optimal cluster separation.
5. **Visualization:** Generate side-by-side scatter plots using `seaborn` and `matplotlib` to compare how well each algorithm separates the wine classes.

## 🚀 How to Run This Project

### Prerequisites
Ensure you have Python installed. You will need the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
