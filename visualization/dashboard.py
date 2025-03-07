import sys
import os
import streamlit as st
import matplotlib.pyplot as plt

# Add the graph folder (which contains main_program.py) to sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
graph_dir = os.path.join(current_dir, "..", "graph")
sys.path.append(graph_dir)

from main_program import (
    query_graph,
    global_monitoring_system,
    plot_alerts,
    # If these functions were originally global, update them to use session_state as needed.
)

# ---------- Initialize Session State ----------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_accounts' not in st.session_state:
    st.session_state.user_accounts = {}  # persistent user data

# ---------- Custom CSS for Dashboard Styling ----------
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #2E3B55;
    }
    .stButton>button {
        background-color: #2E3B55;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("The Guardian of AI Alignment")

# ---------- User Registration and Login ----------
if not st.session_state.logged_in:
    st.header("Login or Register")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            login_username = st.text_input("Email:", key="login_username")
            login_password = st.text_input("Password:", type="password", key="login_password")
            submit_login = st.form_submit_button("Log In")
        if submit_login:
            accounts = st.session_state.user_accounts
            if login_username in accounts and accounts[login_username]["password"] == login_password:
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success("Logged in successfully!")
                st.experimental_rerun()  # Refresh the app to show dashboard
            else:
                st.error("Invalid credentials or user not registered.")
    
    with tab2:
        st.subheader("Register")
        with st.form("registration_form"):
            reg_username = st.text_input("Enter your email:", key="reg_username")
            reg_password = st.text_input("Enter a password:", type="password", key="reg_password")
            submit_reg = st.form_submit_button("Register")
        if submit_reg:
            accounts = st.session_state.user_accounts
            if reg_username in accounts:
                st.warning(f"User '{reg_username}' already exists.")
            else:
                accounts[reg_username] = {"password": reg_password, "projects": []}
                st.success(f"User '{reg_username}' successfully registered.")
                st.info("Please use the Login tab to log in.")
    st.stop()  # Stop further execution until the user logs in.

# ---------- Sign Out Option ----------
if st.session_state.logged_in:
    if st.button("Sign Out"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

# ---------- Customized Dashboard for Logged-In Users ----------
st.header(f"Welcome, {st.session_state.username}!")

# --- Section: Project Submission ---
st.subheader("Submit Your AI Project")
with st.form("project_submission_form"):
    ps_username = st.text_input("Your Email:", key="ps_username", value=st.session_state.username)
    project_desc = st.text_area("Project Description (brief):")
    project_industry = st.text_input("Industry (e.g., Healthcare, Cyber Security):")
    project_source = st.text_input("Project Source (URL or identifier):")
    submit_project_button = st.form_submit_button("Submit Project")
if submit_project_button:
    # Use session_state.user_accounts for persistence
    accounts = st.session_state.user_accounts
    if ps_username in accounts:
        project = {
            "description": project_desc,
            "industry": project_industry.lower(),
            "source": project_source
        }
        accounts[ps_username]["projects"].append(project)
        # Assume check_project_history is defined in main_program.py and works correctly
        history_report = check_project_history(project)
        st.info(f"Project submitted successfully for user '{ps_username}'.\n{history_report}")
    else:
        st.error("User not registered. Please register first.")

# --- Section: View Your Projects ---
st.subheader("Your Submitted Projects")
user_to_view = st.text_input("Enter your email to view your projects:", key="view_user", value=st.session_state.username)
if st.button("Refresh My Projects"):
    st.text(list_projects(user_to_view))

# --- Section: Interactive Graph Query ---
st.header("Ask the Graph for Insights")
user_query = st.text_input("Enter your query for the graph:", key="graph_query")
if st.button("Run Query"):
    result = query_graph(user_query)
    st.subheader("Query Result")
    st.text(result)

# --- Section: Global Monitoring Report ---
st.header("Global Monitoring Report")
if st.button("Refresh Global Report"):
    report = global_monitoring_system()
    st.text(report)
    if "ALERT" in report.upper() or "HIGH-RISK" in report.upper():
        try:
            with open("alarm.mp3", "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        except Exception as e:
            st.error(f"Could not play alarm sound: {e}")

# --- Section: Alerts Dashboard Visualization ---
st.header("Alerts Dashboard")
def get_alerts_figure():
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_alerts()  # This function should generate a plot.
    return fig
st.pyplot(get_alerts_figure())

# --- Section: Popularity Analysis ---
st.header("Popularity Analysis")
refined_query = (
    "Find the ai_node from the 'ai_nodes' collection representing a person, AI, institution, "
    "or company with the highest connectivity (e.g., based on degree or pagerank). Explain why "
    "this entity is considered the most popular."
)
if st.button("Run Popularity Analysis"):
    final_result = query_graph(refined_query)
    st.subheader("Popularity Analysis Result")
    st.text(final_result)

st.text(f"Running with Python executable: {sys.executable}")
