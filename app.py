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

# =========================
# 🔥 CODE GENERATOR FUNCTION
# =========================
def generate_clean_code(model_choice, x_cols, y_col, params):

    if len(x_cols) == 1:
        x_format = f"data['{x_cols[0]}'].values.reshape(-1, 1)"
    else:
        x_format = f"data[{x_cols}]"

    code = f"""
import pandas as pd

# Step 1: Load the dataset
data = pd.read_csv("your_file.csv")

# Step 2: Extract features and target variable
X = {x_format}
y = data['{y_col}'].values
"""

    if model_choice in ["Linear Regression", "Multiple Linear Regression"]:
        code += """
from sklearn.linear_model import LinearRegression

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)
"""

    elif model_choice == "Polynomial Regression":
        code += f"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree={params.get('degree', 3)})
X = poly.fit_transform(X)

model = LinearRegression()
model.fit(X, y)
"""

    elif model_choice == "KNN":
        code += f"""
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors={params.get('k', 3)})
model.fit(X, y)
"""

    elif model_choice == "Decision Tree":
        code += f"""
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth={params.get('depth', 5)},
    criterion='{params.get('criterion', 'entropy')}',
    random_state={params.get('random_state', 42)}
)
model.fit(X, y)
"""

    elif model_choice == "SVM":
        code += f"""
from sklearn.svm import SVC

model = SVC(
    C={params.get('C', 1.0)},
    kernel='{params.get('kernel', 'rbf')}'
)
model.fit(X, y)
"""

    elif model_choice == "Random Forest":
        code += f"""
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators={params.get('n_estimators', 100)},
    criterion='{params.get('criterion', 'entropy')}',
    random_state={params.get('random_state', 42)}
)
model.fit(X, y)
"""

    code += """

# Step 4: Prediction function
def predict(input_value):
    return model.predict([input_value])

# Example
print("Prediction:", predict([5]))
"""

    return code


# =========================
# ⚙️ PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Machine Learning Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Machine Learning Studio")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)

        y_col = st.selectbox("🎯 Target Variable (Y)", df.columns)
        x_cols = st.multiselect("📊 Feature Variables (X)", [col for col in df.columns if col != y_col])

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", value=42)

# =========================
# MAIN APP
# =========================
if file:
    tab_data, tab_model, tab_viz = st.tabs(["Data", "Model", "Visualization"])

    # -------------------------
    # DATA TAB
    # -------------------------
    with tab_data:
        st.dataframe(df)

    # -------------------------
    # MODEL TAB
    # -------------------------
    with tab_model:
        if x_cols:

            model_choice = st.selectbox("Model", [
                "Linear Regression",
                "Multiple Linear Regression",
                "Polynomial Regression",
                "KNN",
                "Decision Tree",
                "SVM",
                "Random Forest"
            ])

            params = {}

            # PARAMETERS
            if model_choice == "Polynomial Regression":
                degree = st.slider("Degree", 2, 5, 3)
                params["degree"] = degree

            elif model_choice == "KNN":
                k = int(np.sqrt(len(df)))
                params["k"] = k
                st.info(f"Auto K = {k}")

            elif model_choice == "Decision Tree":
                depth = 5
                criterion = st.selectbox("Criterion", ["entropy", "gini"])
                params.update({"depth": depth, "criterion": criterion, "random_state": 42})

            elif model_choice == "SVM":
                C = 1.0
                kernel = "rbf"
                params.update({"C": C, "kernel": kernel})

            elif model_choice == "Random Forest":
                params.update({"n_estimators": 100, "criterion": "entropy", "random_state": 42})

            # ERROR HANDLING
            is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
            unique_values = df[y_col].nunique()

            if model_choice in ["KNN", "Decision Tree", "SVM", "Random Forest"]:
                if is_numeric and unique_values > 15:
                    st.error("❌ Classification model with continuous Y")
                    st.stop()

            if model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:
                if not is_numeric:
                    st.error("❌ Regression requires numeric Y")
                    st.stop()

            if st.button("🚀 Train Model"):

                X = pd.get_dummies(df[x_cols])
                y = df[y_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

                # MODEL
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
                    model = DecisionTreeClassifier(**params)

                elif model_choice == "SVM":
                    model = SVC(**params)

                elif model_choice == "Random Forest":
                    model = RandomForestClassifier(**params)

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                st.subheader("📊 Results")

                if model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:
                    st.write("R2:", r2_score(y_test, pred))
                else:
                    st.write("Accuracy:", accuracy_score(y_test, pred))

                # =========================
                # 💻 SHOW CODE BUTTON
                # =========================
                st.divider()
                st.subheader("💻 Generated Python Code")

                if st.button("📜 Show Code"):
                    code = generate_clean_code(model_choice, x_cols, y_col, params)

                    st.code(code, language="python")

                    st.download_button(
                        "📥 Download Code",
                        code,
                        "ml_model_code.py",
                        "text/plain"
                    )

    # -------------------------
    # VISUALIZATION TAB
    # -------------------------
    with tab_viz:
        st.plotly_chart(px.imshow(df.corr(numeric_only=True)))

else:
    st.info("Upload CSV to start")
