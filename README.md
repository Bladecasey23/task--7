# task--7
# Support Vector Machines (SVM) - Binary Classification

## Objective
Use Support Vector Machines (SVMs) to perform both **linear** and **non-linear classification** on a binary dataset.

## Tools & Libraries
- Python
- Scikit-learn
- NumPy
- Matplotlib
- pandas

## Dataset
**Breast Cancer Wisconsin Dataset** 

- Target classes: `M` (Malignant) and `B` (Benign)
- Features: 30 numeric features
- Target column is converted from `'M'` and `'B'` to `1` and `0` respectively

## Steps Followed

### 1. Load and Prepare the Dataset
- Load the Breast Cancer dataset
- Apply `StandardScaler` for feature normalization
- Use PCA to reduce dimensionality for 2D visualization
- Convert target labels `'M'` and `'B'` to numeric values (1 and 0)

### 2. Train SVM Models
- **Linear SVM**: Trained using `SVC(kernel='linear')`
- **RBF SVM**: Trained using `SVC(kernel='rbf')`

### 3. Visualize Decision Boundaries
- Plot decision boundaries using Matplotlib for both kernels
- Use 2D PCA-reduced data for plotting

### 4. Hyperparameter Tuning
- Grid search used to tune `C` and `gamma` for RBF kernel
- Used `GridSearchCV` with 5-fold cross-validation

### 5. Model Evaluation
- Evaluated models using:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
- Visualized best model’s decision boundary

## Results
- Linear and RBF kernels were compared
- Hyperparameter tuning significantly improved RBF model performance
- Final model achieved high accuracy on test set
