import streamlit as st
from collections import Counter, defaultdict

st.set_page_config(page_title="Student Performance Predictor")
st.title("🎓 Student Performance Prediction")
st.write("Bayesian Network Model - Pure Python")

# --- DATASET ---
dataset = [
    ("High", "High", "High", "High", "Excellent"),
    ("High", "High", "Medium", "High", "Good"),
    ("Medium", "High", "High", "Medium", "Good"),
    ("Medium", "Medium", "Medium", "Medium", "Pass"),
    ("Low", "High", "Medium", "Medium", "Pass"),
    ("Low", "Low", "Low", "Low", "Fail"),
    ("Low", "Medium", "Low", "Low", "Fail"),
    ("High", "Medium", "High", "High", "Excellent"),
    ("Medium", "Low", "Medium", "Low", "Fail"),
    ("High", "High", "High", "Medium", "Excellent"),
]

# Train
prior = Counter([r[4] for r in dataset])
total = len(dataset)
prior_prob = {k: v/total for k,v in prior.items()}
likelihood = defaultdict(lambda: defaultdict(Counter))
for s,a,ass,g,p in dataset:
    likelihood['Study'][p][s]+=1
    likelihood['Att'][p][a]+=1
    likelihood['Assign'][p][ass]+=1
    likelihood['Grade'][p][g]+=1

def predict(study, att, assign, grade):
    scores = {}
    for perf in prior:
        p = prior_prob[perf]
        p *= (likelihood['Study'][perf][study] + 1) / (prior[perf] + 3)
        p *= (likelihood['Att'][perf][att] + 1) / (prior[perf] + 3)
        p *= (likelihood['Assign'][perf][assign] + 1) / (prior[perf] + 3)
        p *= (likelihood['Grade'][perf][grade] + 1) / (prior[perf] + 3)
        scores[perf] = p
    s = sum(scores.values())
    for k in scores: scores[k]/=s
    best = max(scores, key=scores.get)
    return best, scores

# --- UI ---
c1,c2 = st.columns(2)
with c1:
    study = st.selectbox("Study Hours", ["Low","Medium","High"])
    att = st.selectbox("Attendance", ["Low","Medium","High"])
with c2:
    assign = st.selectbox("Assignment", ["Low","Medium","High"])
    grade = st.selectbox("Previous Grade", ["Low","Medium","High"])

if st.button("Predict", type="primary"):
    best, probs = predict(study, att, assign, grade)
    st.success(f"Prediction: **{best}**")
    st.write(probs)
    st.bar_chart(probs)
