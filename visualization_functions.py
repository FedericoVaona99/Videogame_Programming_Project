import streamlit as st

# Function to visualize simple text to web page
def write(text):
    st.write(text)

# Function to visualize dataframe to web page
def write_df(df):
    st.dataframe(df)

# Function to create a title to web page
def write_title(text):
    st.title(text)

# Function to create a title to web page
def write_header(text):
    st.header(text)
