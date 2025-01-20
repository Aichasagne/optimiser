import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Charger le dataset
dataset = pd.read_csv('DatasetmalwareExtrait.csv')

# Séparer les features (X) et les labels (Y)
X = dataset[['AddressOfEntryPoint', 'MajorLinkerVersion','MajorImageVersion','MajorOperatingSystemVersion','DllCharacteristics','SizeOfStackReserve','NumberOfSections','ResourceSize']]
Y = dataset[['legitimate']]

# Séparer en données d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Créer une interface Streamlit
st.title("Classification d'Exécution de Logiciels avec Decision Tree")
st.write("Ce modèle permet de prédire si un fichier est légitime ou malveillant.")

# Sélectionner si on utilise le modèle optimisé ou non
use_optimized_model = st.radio("Choisissez le modèle :", ('Sans optimisation', 'Avec optimisation'))

# Entraîner un modèle sans optimisation
if use_optimized_model == 'Sans optimisation':
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    st.write(f"Accuracy (sans optimisation) : {metrics.accuracy_score(Y_test, y_pred):.2f}")
    st.write(f"F-Score (sans optimisation) : {metrics.f1_score(Y_test, y_pred):.2f}")
    st.write(f"Recall (sans optimisation) : {metrics.recall_score(Y_test, y_pred):.2f}")
    st.write("Matrice de confusion (sans optimisation) :")
    st.write(confusion_matrix(Y_test, y_pred))

else:
    # Modèle avec optimisation
    dt_param_grid = {
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    }
    random_search_dt = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions=dt_param_grid,
        n_iter=10,
        scoring="accuracy",
        cv=3,
        random_state=42
    )
    random_search_dt.fit(X_train, Y_train)

    # Meilleur modèle
    best_model = random_search_dt.best_estimator_
    y_pred_optimized = best_model.predict(X_test)

    st.write(f"Meilleurs hyperparamètres : {random_search_dt.best_params_}")
    st.write(f"Accuracy (avec optimisation) : {metrics.accuracy_score(Y_test, y_pred_optimized):.2f}")
    st.write(f"F-Score (avec optimisation) : {metrics.f1_score(Y_test, y_pred_optimized):.2f}")
    st.write(f"Recall (avec optimisation) : {metrics.recall_score(Y_test, y_pred_optimized):.2f}")
    st.write("Matrice de confusion (avec optimisation) :")
    st.write(confusion_matrix(Y_test, y_pred_optimized))
