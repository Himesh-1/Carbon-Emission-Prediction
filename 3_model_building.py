# Predictive data analysis with the Random Forest machine learning algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv('data.csv')

# Exploratory Data Analysis (EDA)
print(data.head())
print(data.info())
print(data.describe())

# Visualize the data
sns.pairplot(data, hue='target')
plt.show()

# Data Preprocessing
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training - Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Model Evaluation
predictions = rf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nAccuracy Score:", accuracy_score(y_test, predictions))

# Feature Importance
feature_importance = pd.DataFrame({'Feature': X.columns,
                                  'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Final Model with Best Parameters
best_rf = grid_search.best_estimator_
best_predictions = best_rf.predict(X_test)

print("\nFinal Model Accuracy:", accuracy_score(y_test, best_predictions))