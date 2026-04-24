import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

=========================

⚙️ PAGE CONFIGURATION

=========================

st.set_page_config(
page_title="Machine Learning Studio",
page_icon="🤖",
layout="wide",
initial_sidebar_state="expanded"
)

st.markdown("""

<style>  
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }  
    h1, h2, h3 { font-weight: 600 !important; }  
</style>  """, unsafe_allow_html=True)

st.title("🤖 Machine Learning Studio")
st.markdown("Upload your dataset, configure your features, and train models in a clean, interactive environment.")

=========================

SIDEBAR

=========================

with st.sidebar:
st.header("1. Upload Data")
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:  
    df = pd.read_csv(file)  
      
    st.divider()  
    st.header("2. Configure Features")  
    columns = df.columns.tolist()  
    y_col = st.selectbox("🎯 Target Variable (Y)", columns)  
    x_cols = st.multiselect("📊 Feature Variables (X)", [col for col in columns if col != y_col])  
      
    st.divider()  
    st.header("3. Train/Test Split")  
    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)  
    random_state = st.number_input("Random State", value=42, step=1)

=========================

MAIN APP

=========================

if file:
tab_data, tab_model, tab_viz = st.tabs([
"🗂️ Data Overview",
"🧠 Model Training & Evaluation",
"📈 Visualizations"
])

# -------------------------  
# DATA TAB  
# -------------------------  
with tab_data:  
    st.subheader("Dataset Preview")  
    st.dataframe(df, use_container_width=True)  
      
    col1, col2 = st.columns(2)  
    col1.metric("Total Rows", df.shape[0])  
    col2.metric("Total Columns", df.shape[1])  
      
    csv = df.to_csv(index=False).encode('utf-8')  
    st.download_button("📥 Download Data", csv, "data.csv")  

# -------------------------  
# MODEL TAB  
# -------------------------  
with tab_model:  
    if x_cols and y_col:  
        st.subheader("Model Selection & Hyperparameters")  

        col_model, col_params = st.columns([1, 2])  

        with col_model:  
            model_choice = st.selectbox(  
                "Choose Algorithm",  
                [  
                    "Linear Regression",  
                    "Multiple Linear Regression",  
                    "Polynomial Regression",  
                    "KNN",  
                    "Decision Tree",  
                    "SVM",  
                    "Random Forest"  
                ]  
            )  

        # =========================  
        # VALIDATION RULES  
        # =========================  
        if model_choice == "Linear Regression" and len(x_cols) != 1:  
            st.warning("⚠️ Linear Regression requires exactly ONE X column")  

        if model_choice == "Multiple Linear Regression" and len(x_cols) < 2:  
            st.warning("⚠️ Multiple Linear Regression requires MULTIPLE X columns")  

        if model_choice == "Polynomial Regression" and len(x_cols) < 1:  
            st.warning("⚠️ Polynomial Regression requires at least ONE X column")  

        # =========================  
        # HYPERPARAMETERS  
        # =========================  
        with col_params:  
            if model_choice == "Polynomial Regression":  
                auto_degree = min(3, len(x_cols)+1)  
                degree = st.slider("Degree", 2, 5, auto_degree)  

            elif model_choice == "KNN":  
                auto_k = int(np.sqrt(len(df)))  
                k = st.slider("K (Neighbors)", 1, 15, auto_k)  

            elif model_choice == "Decision Tree":  
                depth = st.slider("Max Depth", 1, 20, min(5, len(df)//10))  
                criterion = st.selectbox("Criterion", ["entropy", "gini"], index=0)  
                random_state = st.number_input("Random State", value=42)  

            elif model_choice == "SVM":  
                C = st.slider("C Value", 0.1, 10.0, 1.0)  
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)  
                random_state = st.number_input("Random State", value=42)  

            elif model_choice == "Random Forest":  
                n_estimators = st.slider("Estimators", 10, 200, 100)  
                criterion = st.selectbox("Criterion", ["entropy", "gini"], index=0)  
                random_state = st.number_input("Random State", value=42)  

        st.divider()  

        # =========================  
        # ERROR HANDLING  
        # =========================  
        if model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:  
            if not pd.api.types.is_numeric_dtype(df[y_col]):  
                st.error("❌ Y must be numeric for Regression models")  
                st.stop()  

        if any(df[col].dtype == "object" for col in x_cols):  
            st.info("ℹ️ String detected in X → Applying One-Hot Encoding")  

        # =========================  
        # ❗ MODEL-TYPE VALIDATION  
        # =========================  

        # Detect if Y is numeric continuous  
        is_numeric = pd.api.types.is_numeric_dtype(df[y_col])  
        unique_values = df[y_col].nunique()  

        # If classification model but Y is continuous → ERROR  
        if model_choice in ["KNN", "Decision Tree", "SVM", "Random Forest"]:  
            if is_numeric and unique_values > 15:  
                st.error("❌ You selected a Classification model but Y is Continuous.\n👉 Use Regression models instead.")  
                st.stop()  

        # If regression model but Y is categorical → ERROR  
        if model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:  
            if not is_numeric:  
                st.error("❌ You selected a Regression model but Y is Categorical.\n👉 Use Classification models instead.")  
                st.stop()  

        # =========================  
        # TRAIN  
        # =========================  
        if st.button("🚀 Train Model", use_container_width=True):  

            X = pd.get_dummies(df[x_cols])  
            y = df[y_col]  

            X_train, X_test, y_train, y_test = train_test_split(  
                X, y, test_size=test_size, random_state=random_state  
            )  

            # Model creation  
            if model_choice in ["Linear Regression", "Multiple Linear Regression"]:  
                model = LinearRegression()  

            elif model_choice == "Polynomial Regression":  
                poly = PolynomialFeatures(degree=degree)  
                X_train = poly.fit_transform(X_train)  
                X_test = poly.transform(X_test)  
                model = LinearRegression()  

            elif model_choice == "KNN":  
                model = KNeighborsClassifier(n_neighbors=k)  

            elif model_choice == "Decision Tree":  
                model = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=random_state)  

            elif model_choice == "SVM":  
                model = SVC(C=C, kernel=kernel)  

            elif model_choice == "Random Forest":  
                model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=random_state)  

            model.fit(X_train, y_train)  
            pred = model.predict(X_test)  

            st.subheader("📊 Results")  

            if model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:  
                st.metric("R² Score", f"{r2_score(y_test, pred):.4f}")  
                st.metric("MSE", f"{mean_squared_error(y_test, pred):.4f}")  
            else:  
                st.metric("Accuracy", f"{accuracy_score(y_test, pred):.4f}")  
                st.metric("Precision", f"{precision_score(y_test, pred, average='weighted', zero_division=0):.4f}")  
                st.metric("Recall", f"{recall_score(y_test, pred, average='weighted', zero_division=0):.4f}")  
                st.metric("F1 Score", f"{f1_score(y_test, pred, average='weighted', zero_division=0):.4f}")  

                st.write("Confusion Matrix")  
                st.dataframe(pd.DataFrame(confusion_matrix(y_test, pred)))  

    else:  
        st.info("👈 Select X and Y in sidebar")  

# -------------------------  
# VISUALIZATION TAB  
# -------------------------  
with tab_viz:  
    st.subheader("Data Visualizations")  

    with st.expander("🔥 Heatmap", True):  
        fig = px.imshow(df.corr(numeric_only=True), text_auto=True)  
        st.plotly_chart(fig, use_container_width=True)  

    with st.expander("📍 Scatter"):  
        if x_cols:  
            fig = px.scatter(df, x=x_cols[0], y=y_col, color=y_col)  
            st.plotly_chart(fig, use_container_width=True)  

    with st.expander("📊 Histogram"):  
        fig = px.histogram(df, x=y_col)  
        st.plotly_chart(fig, use_container_width=True)

else:
st.info("Upload CSV to start")
