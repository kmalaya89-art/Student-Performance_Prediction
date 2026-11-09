
import streamlit as st
import pandas as pd
import joblib

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# 2. LOAD MODEL AND DATASET
# =====================================

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# =====================================
# 3. DASHBOARD HEADER
# =====================================

st.title("🎓 Student Performance Prediction System")

st.caption(
    "A Data Mining classroom prototype for exploring "
    "student academic risk using machine learning."
)

st.info(
    "This system provides experimental predictions "
    "based on historical student records. "
    "It must not be used as the sole basis for student decisions."
)

# =====================================
# 4. DASHBOARD METRICS
# =====================================

c1, c2, c3 = st.columns(3)

c1.metric("Total Student Records", len(data))

c2.metric(
    "Average Final Grade",
    round(data["G3"].mean(), 2)
)

c3.metric(
    "Historical At-Risk Records",
    int((data["G3"] < 10).sum())
)

# =====================================
# 5. DATASET PREVIEW
# =====================================

st.subheader("📊 Dataset Overview")

st.write(
    "Preview of the historical Mathematics student records."
)

st.dataframe(
    data.head(10),
    use_container_width=True
)

# =====================================
# 6. MODEL EVALUATION
# =====================================

st.subheader("🤖 Model Information")

st.write("Model: Random Forest Classifier")
st.write("Training dataset: UCI Student Performance")
st.write("Prediction target: G3 < 10 = AT RISK")

# =====================================
# 7. SAMPLE INPUT DOWNLOAD
# =====================================

st.subheader("📁 Student CSV Prediction")

st.write(
    "Download the sample input CSV and use it as "
    "a template for your prediction file."
)

required_columns = list(model.feature_names_in_)

template = data[required_columns].head(5)

st.download_button(
    label="⬇️ Download Sample Input CSV",
    data=template.to_csv(index=False),
    file_name="sample_prediction_input.csv",
    mime="text/csv"
)

# =====================================
# 8. UPLOAD STUDENT CSV
# =====================================

uploaded = st.file_uploader(
    "Upload Student CSV File",
    type=["csv"]
)

if uploaded is not None:

    try:
        new_data = pd.read_csv(uploaded)

        st.subheader("Uploaded Data Preview")
        st.dataframe(
            new_data.head(),
            use_container_width=True
        )

        missing = [
            col for col in required_columns
            if col not in new_data.columns
        ]

        if missing:
            st.error(
                "Missing required columns: "
                + ", ".join(missing)
            )

        elif new_data.empty:
            st.warning("The uploaded CSV contains no records.")

        else:
            # Prepare input using training column order
            prediction_input = new_data[required_columns]

            # Generate predictions
            predictions = model.predict(prediction_input)

            results = new_data.copy()

            results["Prediction"] = [
                "AT RISK" if value == 1
                else "NOT AT RISK"
                for value in predictions
            ]

            # Display results
            st.subheader("📈 Prediction Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            at_risk = int((predictions == 1).sum())
            not_at_risk = int((predictions == 0).sum())

            col1, col2 = st.columns(2)

            col1.metric("Predicted At Risk", at_risk)
            col2.metric("Predicted Not At Risk", not_at_risk)

            # Download predictions
            st.download_button(
                label="⬇️ Download Prediction Results",
                data=results.to_csv(index=False),
                file_name="student_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Unable to process uploaded CSV: {e}")

# =====================================
# 9. FOOTER
# =====================================

st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Data Mining Laboratory | Random Forest Classifier"
)
