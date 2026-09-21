import streamlit as st
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.subheader("Using Bayesian Network Model")

# --- DATASET ---
# Replace this with pd.read_csv("your_data.csv")
data = pd.DataFrame({
    'Study_Hours': ['Low', 'Medium', 'High', 'High', 'Medium', 'Low', 'High', 'Medium', 'High', 'Low'],
    'Attendance': ['Low', 'High', 'High', 'Medium', 'Low', 'High', 'High', 'High', 'Low', 'Medium'],
    'Assignment': ['Low', 'Medium', 'High', 'High', 'Low', 'Medium', 'High', 'Medium', 'High', 'Low'],
    'Previous_Grade': ['Low', 'Medium', 'High', 'High', 'Medium', 'Low', 'High', 'Medium', 'High', 'Low'],
    'Performance': ['Fail', 'Pass', 'Excellent', 'Good', 'Fail', 'Pass', 'Excellent', 'Good', 'Good', 'Fail']
})

@st.cache_resource
def train_model():
    model = BayesianNetwork([
        ('Study_Hours', 'Performance'),
        ('Attendance', 'Performance'),
        ('Assignment', 'Performance'),
        ('Previous_Grade', 'Performance')
    ])
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

model = train_model()
infer = VariableElimination(model)

# --- WEBSITE UI ---
col1, col2 = st.columns(2)
with col1:
    study = st.selectbox("Study Hours Per Day", ["Low (<2h)", "Medium (2-4h)", "High (>4h)"])
    study_val = study.split()[0]
    attendance = st.selectbox("Attendance %", ["Low (<60%)", "Medium (60-80%)", "High (>80%)"])
    att_val = attendance.split()[0]

with col2:
    assignment = st.selectbox("Assignment Completion", ["Low", "Medium", "High"])
    grade = st.selectbox("Previous Grade", ["Low", "Medium", "High"])

if st.button("Predict Performance", type="primary"):
    evidence = {
        'Study_Hours': study_val,
        'Attendance': att_val,
        'Assignment': assignment,
        'Previous_Grade': grade
    }
    result = infer.query(variables=['Performance'], evidence=evidence)

    # Get probabilities
    states = result.state_names['Performance']
    probs = result.values

    best_idx = probs.argmax()
    best_state = states[best_idx]
    confidence = probs[best_idx] * 100

    st.success(f"**Predicted Result: {best_state}** ({confidence:.1f}% confidence)")

    st.write("Full Probability Distribution:")
    prob_df = pd.DataFrame({"Performance": states, "Probability": probs})
    st.bar_chart(prob_df.set_index("Performance"))

st.divider()
st.caption("Model: Bayesian Network | P(Performance | Study, Attendance, Assignment, Grade)")