import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Student Support Predictor", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #dcecf2 0%, #cfe3db 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f4e5f 0%, #2c786c 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        color: #12343b;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .section-card {
        background: linear-gradient(135deg, #dff3ea 0%, #cfe8de 100%);
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        border: 1px solid rgba(31, 78, 95, 0.12);
    }

    .about-card {
        background: linear-gradient(135deg, #dff3ea 0%, #cfe8de 100%);
        padding: 28px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(18, 52, 59, 0.12);
        border: 1px solid rgba(31, 78, 95, 0.12);
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2c786c 0%, #1f4e5f 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #245f56 0%, #173943 100%);
        color: white;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #ff9f1c 0%, #ffbf69 100%);
        color: #1f1f1f;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.3rem;
        font-weight: 700;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(90deg, #f08c00 0%, #ffb347 100%);
        color: #1f1f1f;
    }

    .small-note {
        color: #1f3b3d;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About Me"])

FIELD_LABELS = {
    "school": "School",
    "sex": "Gender",
    "age": "Age (years)",
    "famsize": "Family Size",
    "Medu": "Mother's Education Level",
    "Mjob": "Mother's Job",
    "Fjob": "Father's Job",
    "guardian": "Guardian",
    "schoolsup": "School Support",
    "famsup": "Family Support",
    "paid": "Extra Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery School",
    "internet": "Internet Access at Home",
    "studytime": "Weekly Study Time",
    "failures": "Number of Past Class Failures",
    "freetime": "Free Time After School",
    "health": "Current Health Status",
    "absences": "Number of School Absences (days)",
    "G1": "First Period Grade (0 to 20 points)",
    "G2": "Second Period Grade (0 to 20 points)",
}


def pretty_label(column_name):
    return FIELD_LABELS.get(column_name, column_name)


def show_about_page():
    st.markdown('<div class="main-title">About Me</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="about-card">
            <h2>Hussein Sabbagh</h2>
            <p class="small-note">
                Hello! I'm Hussein Sabbagh, a Machine Learning enthusiast.
            </p>
            <p class="small-note">
                This dashboard allows users to upload a student performance dataset, explore the data,
                train a machine learning model, and predict whether a student may need extra academic support.
            </p>
            <p class="small-note">
                The goal is to show how machine learning can be used in education in a simple,
                interactive, and understandable way.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def load_student_csv(uploaded_file):
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=";")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    if len(df.columns) == 1:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        df.columns = df.columns.str.replace(
            "\ufeff", "", regex=False).str.strip()

    return df


def show_prediction_page():
    st.markdown('<div class="main-title">Student Support Predictor</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            Upload the student performance CSV file to train the model and start prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Browse and upload the CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload the CSV file first.")
        return

    try:
        df = load_student_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.markdown('</div>', unsafe_allow_html=True)

    if "G3" not in df.columns:
        st.error(
            "The uploaded file was read successfully, but the Final Grade column was not found.")
        st.write("Detected columns:", list(df.columns))
        return

    threshold = st.slider(
        "Final grade threshold used to mark a student as needing support",
        min_value=0,
        max_value=20,
        value=10
    )

    df = df.dropna(subset=["G3"]).copy()
    df["needs_support"] = (df["G3"] <= threshold).astype(int)

    if df["needs_support"].nunique() < 2:
        st.error(
            "The selected threshold creates only one class. Change the threshold.")
        return

    columns_to_remove = [
        "G3",
        "needs_support",
        "address",
        "Pstatus",
        "Fedu",
        "reason",
        "traveltime",
        "higher",
        "romantic",
        "famrel",
        "goout",
        "Dalc",
        "Walc",
    ]

    X = df.drop(columns=columns_to_remove, errors="ignore")
    y = df["needs_support"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return

    st.success("Model trained successfully.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Enter Student Information")
    st.info(
        "Units: Age is measured in years, absences are measured in days, and grades are scored from 0 to 20 points."
    )

    user_input = {}

    with st.form("prediction_form"):
        for col in X.columns:
            label = pretty_label(col)

            if col in categorical_cols:
                options = sorted(
                    df[col].dropna().astype(str).unique().tolist())
                user_input[col] = st.selectbox(label, options)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                default_val = float(df[col].median())
                user_input[col] = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )

        submitted = st.form_submit_button("Start Prediction")

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.warning(
                    "Prediction: This student may need extra academic support.")
            else:
                st.success(
                    "Prediction: This student is likely not at immediate academic risk.")

            st.markdown(
                f"""
                <div class="section-card">
                    <h3>Prediction Result</h3>
                    <p class="small-note">Support risk score: <strong>{probability:.2%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if page == "Prediction":
    show_prediction_page()
else:
    show_about_page()
