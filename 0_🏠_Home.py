import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("# Employee Attrition and Emotion Detection")

st.sidebar.success("Select a Page above.")
st.write('This app predicts employee attrition')
st.image('home.png',width=900)
