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
# ⚙️ PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Machine Learning Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

for key, val in {
    "degree": 3,
    "k": 5,
    "depth": 5,
    "C": 1.0,
    "n_estimators": 100
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

criterion = "entropy"
kernel = "rbf"

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Machine Learning Studio")
st.markdown("Upload your dataset, configure your features, and train models in a clean, interactive environment.")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("1. Upload Data")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if "model_choice" not in st.session_state:
        st.session_state.model_choice = None
    if "pred" not in st.session_state:
        st.session_state.pred = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None

    # =========================
    # COLUMN CLEANING FUNCTION
    # =========================
    def clean_columns(df):
        df = df.copy()

        # Remove unnamed/artifact columns
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        # Remove constant columns
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index
        df = df.drop(columns=constant_cols)

        # Clean column names
        df.columns = df.columns.str.strip()

        return df

    if file:
        raw_df = pd.read_csv(file)

        # 🔥 CLEAN ON LOAD (important)
        df = clean_columns(raw_df)

        # Store original + cleaned separately (best practice)
        st.session_state.raw_df = raw_df
        st.session_state.df = df

        st.success("✅ Data loaded & basic cleaning applied")

        st.divider()

        # =========================
        # FEATURE CONFIG
        # =========================
        st.header("2. Configure Features")

        columns = df.columns.tolist()

        y_col = st.selectbox("🎯 Target Variable (Y)", columns, key="y_col")

        x_cols = st.multiselect(
            "📊 Feature Variables (X)",
            [col for col in columns if col != y_col],
            key="x_cols"
        )

        st.divider()

        # =========================
        # TRAIN TEST SPLIT
        # =========================
        st.header("3. Train/Test Split")

        test_size = st.slider(
            "Test Size Ratio",
            0.1, 0.5, 0.2, 0.05,
            key="test_size"
        )

        random_state = st.number_input(
            "Random State",
            value=42,
            step=1,
            key="random_state"
        )
def find_best_param(model_type, X_train, X_test, y_train, y_test):

    best_param = None
    best_score = -float("inf")

    results = []

    if model_type == "Polynomial":
        for d in range(1, 21):
            poly = PolynomialFeatures(degree=d)
            X_train_p = poly.fit_transform(X_train)
            X_test_p = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_p, y_train)
            pred = model.predict(X_test_p)

            score = -mean_squared_error(y_test, pred)  # minimize MSE

            results.append((d, score))

            if score > best_score:
                best_score = score
                best_param = d

    elif model_type == "KNN":
        for k in range(1, 16):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            score = accuracy_score(y_test, pred)

            results.append((k, score))

            if score > best_score:
                best_score = score
                best_param = k

    elif model_type == "Decision Tree":
        for d in range(1, 21):
            model = DecisionTreeClassifier(max_depth=d, random_state=random_state, criterion=criterion)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            score = accuracy_score(y_test, pred)

            results.append((d, score))

            if score > best_score:
                best_score = score
                best_param = d

    elif model_type == "SVM":
        for c in range(1, 100):
            model = SVC(C=c, kernel=kernel, random_state=random_state)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            score = accuracy_score(y_test, pred)

            results.append((c, score))

            if score > best_score:
                best_score = score
                best_param = c

    elif model_type == "Random Forest":
        for n in range(1, 100, 1):
            model = RandomForestClassifier(n_estimators=n,random_state=random_state,criterion=criterion)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            score = accuracy_score(y_test, pred)

            results.append((n, score))

            if score > best_score:
                best_score = score
                best_param = n

    return best_param, results

# =========================
# MAIN APP
# =========================
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None
if file:
    tab_data, tab_model, tab_viz,tab_code = st.tabs([
        "🗂️ Data Overview", 
        "🧠 Model Training & Evaluation", 
        "📈 Visualizations",
        "👨‍💻 Generate Code"
    ])

    

    # -------------------------
    # DATA TAB
    # -------------------------
    with tab_data:

        # =========================
        # COLUMN CLEANING FUNCTION
        # =========================
        def clean_columns(df):
            df = df.copy()

            # Remove unnamed / artifact columns
            df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

            # Remove completely empty columns
            df = df.dropna(axis=1, how="all")

            # Remove constant columns
            nunique = df.nunique()
            constant_cols = nunique[nunique <= 1].index
            df = df.drop(columns=constant_cols)

            # Clean column names
            df.columns = df.columns.str.strip()

            return df

        # =========================
        # INIT CLEAN DATA
        # =========================
        if "clean_df" not in st.session_state:
            cleaned = clean_columns(df)
            st.session_state.clean_df = cleaned

        clean_df = st.session_state.clean_df

        # =========================
        # DATA PREVIEW
        # =========================
        st.subheader("Dataset Preview (Cleaned Data)")

        colR1, colR2 = st.columns([1, 5])
        with colR1:
            if st.button("🔄 Refresh", key="refresh_btn"):
                st.rerun()

        st.dataframe(clean_df, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Rows", clean_df.shape[0])
        col2.metric("Total Columns", clean_df.shape[1])

        # =========================
        # MISSING VALUES INFO
        # =========================
        st.subheader("🔍 Missing Values Overview")
        st.dataframe(clean_df.isna().sum())

        # =========================
        # DATA CLEANING OPTIONS
        # =========================
        st.subheader("🧹 Data Cleaning Options")

        colA, colB = st.columns(2)

        # -------------------------
        # REMOVE NaN ROWS
        # -------------------------
        with colA:
            if st.button("🗑️ Remove NaN Rows", key="remove_nan"):
                st.session_state.clean_df = clean_df.dropna()
                st.success("Rows with NaN removed")
                st.rerun()

        # -------------------------
        # RESET DATA
        # -------------------------
        with colB:
            if st.button("🔄 Reset Data", key="reset_data"):
                st.session_state.clean_df = clean_columns(df)
                st.success("Data reset & cleaned")
                st.rerun()

        st.divider()

        # =========================
        # NUMERIC NaN HANDLING
        # =========================
        numeric_cols = clean_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:
            st.subheader("📊 Handle Numeric NaN")

            num_method = st.selectbox(
                "Select Method",
                ["Mean", "Median", "Zero"],
                key="num_method"
            )

            if st.button("Apply Numeric Fill", key="num_fill"):
                if num_method == "Mean":
                    clean_df[numeric_cols] = clean_df[numeric_cols].fillna(clean_df[numeric_cols].mean())
                elif num_method == "Median":
                    clean_df[numeric_cols] = clean_df[numeric_cols].fillna(clean_df[numeric_cols].median())
                else:
                    clean_df[numeric_cols] = clean_df[numeric_cols].fillna(0)

                st.session_state.clean_df = clean_df
                st.success("Numeric NaN handled")
                st.rerun()

        # =========================
        # CATEGORICAL NaN HANDLING
        # =========================
        cat_cols = clean_df.select_dtypes(include="object").columns

        if len(cat_cols) > 0:
            st.subheader("🧩 Handle Categorical NaN")

            cat_method = st.selectbox(
                "Select Method",
                ["Mode", "Custom Value"],
                key="cat_method"
            )

            custom_value = ""
            if cat_method == "Custom Value":
                custom_value = st.text_input("Enter Custom Value", key="custom_val")

            if st.button("Apply Categorical Fill", key="cat_fill"):
                for col in cat_cols:
                    if cat_method == "Mode":
                        clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])
                    else:
                        clean_df[col] = clean_df[col].fillna(custom_value)

                st.session_state.clean_df = clean_df
                st.success("Categorical NaN handled")
                st.rerun()

        st.divider()

        # =========================
        # DOWNLOAD CLEAN DATA
        # =========================
        csv = clean_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Cleaned Data",
            csv,
            "cleaned_data(By Ayush H.Kadiya).csv",
            key="download_clean"
        )
    # -------------------------
    # MODEL TAB
    # -------------------------
    with tab_model:
        if x_cols and y_col:
            X = pd.get_dummies(df[x_cols])
            y = df[y_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            st.subheader("Model Selection & Hyperparameters")

            col_model, col_params = st.columns([1, 2])

            with col_model:
                st.session_state.model_choice = st.selectbox(
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
                model_choice = st.session_state.model_choice

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
                    # X.train=X_train.get_dummies(X_train)
                    # X.test=X_test.get_dummies(X_test)
                    if st.button("🔍 Find Best Degree"):
                        best_degree, results = find_best_param(
                            "Polynomial", X_train, X_test, y_train, y_test
                        )

                        st.session_state.degree = best_degree   # 🔥 IMPORTANT

                        # for d, score in results:
                        #     st.write(f"Degree {d} → Score: {score:.4f}")

                        st.success(f"Best Degree: {best_degree}")

                    degree = st.slider(
                        "Degree",
                        1,
                        21,
                        key="degree"   # 🔥 bind directly
                    )
                elif model_choice == "KNN":

                    if st.button("🔍 Find Best K"):
                        best_k, _ = find_best_param("KNN", X_train, X_test, y_train, y_test)
                        st.session_state.k = best_k   # 🔥

                    k = st.slider("K", 1, 16, key="k")

                elif model_choice == "Decision Tree":

                    criterion = st.selectbox(
                        "Criterion",
                        ["gini", "entropy"],
                        key="dt_criterion"
                    )

                    if st.button("🔍 Find Best Depth"):
                        best_depth, _ = find_best_param("Decision Tree", X_train, X_test, y_train, y_test)
                        st.session_state.depth = best_depth

                    depth = st.slider("Depth", 1, 21, key="depth")

                elif model_choice == "SVM":

                    kernel = st.selectbox(
                        "Kernel",
                        ["rbf", "linear", "poly"],
                        key="svm_kernel"
                    )

                    if st.button("🔍 Find Best C"):
                        best_C, _ = find_best_param("SVM", X_train, X_test, y_train, y_test)
                        st.session_state.C = best_C

                    C = st.slider("C", 1, 100, key="C")

                elif model_choice == "Random Forest":

                    criterion = st.selectbox(
                        "Criterion",
                        ["gini", "entropy"],
                        key="rf_criterion"
                    )

                    if st.button("🔍 Find Best Estimators"):
                        best_n, _ = find_best_param("Random Forest", X_train, X_test, y_train, y_test)
                        st.session_state.n_estimators = best_n

                    n_estimators = st.slider("Estimators", 1, 100, key="n_estimators")
            st.divider()

            # =========================
            # ERROR HANDLING
            # =========================
            if model_choice and model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:
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
            if model_choice and model_choice in ["KNN", "Decision Tree", "SVM", "Random Forest"]:
                if is_numeric and unique_values > 15:
                    st.error("❌ You selected a Classification model but Y is Continuous.\n👉 Use Regression models instead.")
                    st.stop()

            # If regression model but Y is categorical → ERROR
            if model_choice and model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:
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
                if model_choice and model_choice in ["Linear Regression", "Multiple Linear Regression"]:
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
                st.session_state.pred = model.predict(X_test)
                st.session_state.y_test = y_test
                pred = st.session_state.pred
                y_test = st.session_state.y_test

                st.subheader("📊 Results")

                if model_choice and model_choice in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:
                    st.metric("R² Score", f"{r2_score(y_test, pred):.4f}")
                    st.metric("MSE", f"{mean_squared_error(y_test, pred):.4f}")
                else:
                    st.metric("Accuracy", f"{accuracy_score(y_test, pred):.4f}")
                    st.metric("Precision", f"{precision_score(y_test, pred, average='weighted', zero_division=0):.4f}")
                    st.metric("Recall", f"{recall_score(y_test, pred, average='weighted', zero_division=0):.4f}")
                    st.metric("F1 Score", f"{f1_score(y_test, pred, average='weighted', zero_division=0):.4f}")

                    st.write("Confusion Matrix")
                    cm = confusion_matrix(y_test, pred)
                    st.dataframe(pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test)))

        else:
            st.info("👈 Select X and Y in sidebar")

        # -------------------------
        # VISUALIZATION TAB
        # -------------------------
    with tab_viz:
        model_choice = st.session_state.get("model_choice", None)
        pred = st.session_state.get("pred", None)
        y_test = st.session_state.get("y_test", None)
        st.subheader("📊 Data Visualizations")

        # =========================
        # 🔥 HEATMAP
        # =========================
        with st.expander("🔥 Correlation Heatmap", True):
            numeric_df = df.select_dtypes(include=np.number)

            if numeric_df.shape[1] >= 2:
                if model_choice and pred is not None and y_test is not None:
                    if model_choice not in ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]:
                        h = confusion_matrix(y_test, pred)
                    else:
                        h = numeric_df.corr()
                else:
                    h = numeric_df.corr()
                fig = px.imshow(
                    h,
                    # confusion_matrix(y_test, pred),
                    text_auto=True,
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for heatmap")

        # =========================
        # 📍 SCATTER PLOT (Dynamic)
        # =========================
        with st.expander("📍 Scatter Plot"):
            if len(df.columns) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    scatter_x = st.selectbox("Select X-axis", df.columns, key="scatter_x")

                with col2:
                    scatter_y = st.selectbox("Select Y-axis", df.columns, index=1, key="scatter_y")

                color_option = st.selectbox("Color By (Optional)", [None] + df.columns.tolist())

                fig = px.scatter(
                    df,
                    x=scatter_x,
                    y=scatter_y,
                    color=color_option if color_option else None
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dataset needs at least 2 columns")

        # =========================
        # 📊 HISTOGRAM (Dynamic)
        # =========================
        with st.expander("📊 Histogram"):
            hist_col = st.selectbox("Select Column", df.columns, key="hist_col")

            bins = st.slider("Number of Bins", 5, 100, 20)

            fig = px.histogram(
                df,
                x=hist_col,
                nbins=bins
            )
            st.plotly_chart(fig, use_container_width=True)

        # =========================
        # ➕ BONUS: BOX PLOT
        # =========================
        with st.expander("📦 Box Plot (Outliers Detection)"):
            box_col = st.selectbox("Select Column for Box Plot", df.columns, key="box_col")

            fig = px.box(df, y=box_col)
            st.plotly_chart(fig, use_container_width=True)
    with tab_code:
        # =========================
        # 💻 GENERATE CODE BUTTON
        # =========================
        if st.button("💻 Generate Code", use_container_width=True):

            file_name = file.name if file else "data(By Ayush H.Kadiya).csv"

            x_code = f"{x_cols}" if len(x_cols) > 1 else f'["{x_cols[0]}"]'
            y_code = f'"{y_col}"'

            base_code = f"""
import pandas as pd
from sklearn.model_selection import train_test_split

#Please clean  Data before using this code by your self and also make sure to install all the required libraries before running this code
        """

            model_code = ""
            metrics_code = ""

            # =========================
            # MODEL-SPECIFIC CODE
            # =========================
            if model_choice and model_choice in ["Linear Regression", "Multiple Linear Regression"]:
                model_code = f"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("{file_name}")
X = df[{x_code}]
y = df[{y_code}]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
        """

            elif model_choice == "Polynomial Regression":
                model_code = f"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("{file_name}")
X = df[{x_code}]
y = df[{y_code}]

X = pd.get_dummies(X)

poly = PolynomialFeatures(degree={degree})
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
        """

            elif model_choice == "KNN":
                model_code = f"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("{file_name}")
X = df[{x_code}]
y = df[{y_code}]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})

model = KNeighborsClassifier(n_neighbors={k})
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
        """

            elif model_choice == "Decision Tree":
                model_code = f"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("{file_name}")
X = df[{x_code}]
y = df[{y_code}]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})

model = DecisionTreeClassifier(max_depth={depth}, criterion="{criterion}", random_state={random_state})
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
        """

            elif model_choice == "SVM":
                model_code = f"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("{file_name}")
X = df[{x_code}]
y = df[{y_code}]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})

model = SVC(C={C}, kernel="{kernel}")
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
        """

            elif model_choice == "Random Forest":
                model_code = f"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("{file_name}")
X = df[{x_code}]
y = df[{y_code}]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})

model = RandomForestClassifier(n_estimators={n_estimators}, criterion="{criterion}", random_state={random_state})
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
        """

            final_code = base_code + model_code

            st.subheader("📄 Generated Python Code")
            st.code(final_code, language="python")

            st.download_button(
                label="📥 Download Code",
                data=final_code,
                file_name="generated_model(by Ayush H.Kadiya).py",
                mime="text/plain"
            )
        else:
            st.info("Upload CSV to start")
            
