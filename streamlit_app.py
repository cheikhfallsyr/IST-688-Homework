import streamlit as st

st.title("HW Manager")

hw1 = st.Page("HW/HW1.py", title="HW 1")
hw2 = st.Page("HW/HW2.py", title="HW 2", default=True)

selected_page = st.navigation([hw1, hw2])
selected_page.run()