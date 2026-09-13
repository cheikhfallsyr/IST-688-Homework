import streamlit as st

st.title("HW Manager")

hw1 = st.Page("HW/HW1.py", title="HW 1")
hw2 = st.Page("HW/HW2.py", title="HW 2")
hw3 = st.Page("HW/HW3.py", title="HW 3", default=True)



selected_page = st.navigation([hw1, hw2, hw3])
selected_page.run()