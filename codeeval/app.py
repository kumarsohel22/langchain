import streamlit as st
import pandas as pd
import json
from code_eval import run_code, evaluate_code, generate_test_cases, LANGUAGES

# Page config
st.set_page_config(page_title="AI Code Evaluator", page_icon="💻", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stTextArea textarea {font-family: monospace; font-size: 14px;}
    .pass-badge {background-color: #4CAF50; color: white; padding: 3px 8px; border-radius: 5px;}
    .fail-badge {background-color: #F44336; color: white; padding: 3px 8px; border-radius: 5px;}
    .score-box {background: #f9f9f9; padding: 10px; border-radius: 8px; margin-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💻 AI Code Evaluator & Test Case Generator")
st.write("Run, evaluate, and test your code across multiple languages with AI-powered insights.")

# Inputs
problem_statement = st.text_area("📝 Problem Statement", height=120)
language = st.selectbox("Select Language", list(LANGUAGES.keys()))
code = st.text_area("💻 Your Code", height=250, placeholder="Write your code here...")
stdin_input = st.text_area("📥 Sample Input (Optional)", height=80)

col1, col2, col3 = st.columns(3)
with col1:
    run_btn = st.button("▶ Run Code")
with col2:
    eval_btn = st.button("📊 Evaluate Code")
with col3:
    test_btn = st.button("🧪 Generate and Run Test Cases")

# Actions
if run_btn and code:
    output = run_code(language, code, stdin_input)
    st.subheader("Execution Output")
    st.code(output)

if eval_btn and code:
    with st.spinner("Evaluating code..."):
        evaluation = evaluate_code(problem_statement, code, stdin_input)
    st.subheader("📊 Code Evaluation")
    for metric, details in evaluation.items():
        st.markdown(f"**{metric}**")
        st.markdown(f"<div class='score-box'>Score: {details['score']}<br>{details['explanation']}</div>", unsafe_allow_html=True)

if test_btn and code:
    with st.spinner("Generating test cases..."):
        results, inputs, evaluation = generate_test_cases(problem_statement, code, language, stdin_input)

    # Parse results into a broader DataFrame
    test_data = []
    for res in results:
        # Expected format from generate_test_cases: "Input: X | Expected: Y | Status"
        parts = res.split("|")
        input_val = parts[0].replace("Input:", "").strip()
        expected_val = parts[1].replace("Expected:", "").strip()
        status_part = parts[2].strip()

        if "Pass" in status_part:
            status_html = '<span class="pass-badge">✅ Pass</span>'
            actual_val = expected_val
            remarks = "Output matches expected"
        else:
            status_html = '<span class="fail-badge">❌ Fail</span>'
            # Extract actual output from fail text
            actual_val = status_part.split("Got:")[-1].strip().replace(")", "")
            remarks = f"Expected '{expected_val}' but got '{actual_val}'"

        test_data.append({
            "Input": input_val,
            "Expected Output": expected_val,
            "Actual Output": actual_val,
            "Status": status_html,
            "Remarks": remarks
        })

    df = pd.DataFrame(test_data)

    # Display table with HTML badges
    st.subheader("🧪 Test Case Results")
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

