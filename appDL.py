import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, r2_score, mean_absolute_error)

st.set_page_config(page_title="🧠 MLP Application", layout="wide")
st.title("🔍 Application MLP Classifier & Regressor by Asmaa Faris")

# Base brute
st.header("1️⃣ Base de données brute")
brut_file = st.file_uploader("📥 Importez la base brute (.xlsx)", type=["xlsx"], key="brut")
df_brut = pd.read_excel(brut_file) if brut_file else None

if df_brut is not None:
    st.dataframe(df_brut.head())

# Base nettoyée
st.header("2️⃣ Base nettoyée pour apprentissage")
clean_file = st.file_uploader("📥 Importez la base nettoyée (.xlsx)", type=["xlsx"], key="clean")
df_clean = pd.read_excel(clean_file) if clean_file else None

if df_clean is not None:
    st.dataframe(df_clean.head())

    all_cols = df_clean.columns.tolist()
    target = st.selectbox("🎯 Variable cible :", all_cols)
    features = st.multiselect("📌 Variables explicatives :", [c for c in all_cols if c != target])

    mode = st.sidebar.radio("🧠 Choix du modèle :", ["Classification", "Régression"])

    st.sidebar.header("⚙️ Hyperparamètres")
    hidden_layer_sizes = st.sidebar.text_input("Couches cachées (ex: 100,50)", value="100,50")
    activation = st.sidebar.selectbox("Fonction d'activation", ["relu", "tanh", "logistic"])
    max_iter = st.sidebar.slider("Itérations max", 100, 2000, 500)

    hidden_layers = tuple(int(x.strip()) for x in hidden_layer_sizes.split(","))

    if features and target:
        X = df_clean[features].copy()
        y = df_clean[target].copy()

        # Encodage
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        if mode == "Classification":
            model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=max_iter)
        else:
            model = MLPRegressor(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=max_iter)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        st.header("3️⃣ Résultats du modèle")
        if mode == "Classification":
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Train": [
                    accuracy_score(y_train, y_train_pred),
                    precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    recall_score(y_train, y_train_pred, average='weighted'),
                    f1_score(y_train, y_train_pred, average='weighted')
                ],
                "Test": [
                    accuracy_score(y_test, y_test_pred),
                    precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    recall_score(y_test, y_test_pred, average='weighted'),
                    f1_score(y_test, y_test_pred, average='weighted')
                ]
            })
        else:
            metrics_df = pd.DataFrame({
                "Metric": ["MSE", "MAE", "R2 Score"],
                "Train": [
                    mean_squared_error(y_train, y_train_pred),
                    mean_absolute_error(y_train, y_train_pred),
                    r2_score(y_train, y_train_pred)
                ],
                "Test": [
                    mean_squared_error(y_test, y_test_pred),
                    mean_absolute_error(y_test, y_test_pred),
                    r2_score(y_test, y_test_pred)
                ]
            })
        st.dataframe(metrics_df)

        # Architecture
        st.subheader("🧬 Architecture du réseau de neurones")
        fig, ax = plt.subplots(figsize=(10, 3))
        input_neurons = len(features)
        hidden_neurons = sum(hidden_layers)
        output_neurons = 1 if len(np.unique(y)) == 2 or mode == "Régression" else len(np.unique(y))

        def draw_layer(neurons, layer_x, color):
            for i in range(neurons):
                ax.scatter(layer_x, -i, s=600, edgecolors='black', facecolors=color)

        # Inputs
        draw_layer(input_neurons, 0, "skyblue")
        # Hidden
        pos = 1
        for layer_size in hidden_layers:
            draw_layer(layer_size, pos, "orange")
            pos += 1
        # Output
        draw_layer(output_neurons, pos, "lightgreen")

        # Arrows
        for i in range(input_neurons):
            for j in range(hidden_layers[0]):
                ax.annotate('', xy=(1, -j), xytext=(0, -i), arrowprops=dict(arrowstyle="->"))
        for j in range(hidden_layers[0]):
            for k in range(output_neurons):
                ax.annotate('', xy=(pos, -k), xytext=(1, -j), arrowprops=dict(arrowstyle="->"))

        ax.axis('off')
        st.pyplot(fig)

        # Prédiction manuelle
        if df_brut is not None:
            st.header("4️⃣ Prédiction manuelle à partir de la base brute")
            manual_input = []
            st.subheader("🖊️ Saisissez les valeurs d’entrée :")
            for col in features:
                if col in df_brut.select_dtypes(include='object').columns:
                    val = st.selectbox(f"{col} (catégorielle)", df_brut[col].unique())
                    encoder = LabelEncoder()
                    encoder.fit(df_brut[col].astype(str))
                    val = encoder.transform([val])[0]
                    manual_input.append(val)
                else:
                    val = st.number_input(f"{col} (numérique)", value=float(df_brut[col].mean()))
                    manual_input.append(val)

            input_scaled = scaler.transform([manual_input])
            if st.button("🔮 Prédire"):
                pred = model.predict(input_scaled)[0]
                st.success(f"🎯 Résultat de la prédiction : **{pred}**")
                if mode == "Classification" and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)[0]
                    st.info(f"🔢 Probabilités : {np.round(proba, 3)}")
