# Videogames Programming Project

import pandas as pd
import numpy as np
import visualization_functions as vis
import streamlit as st
import matplotlib.pyplot as plt

#####################
# EDA
#####################

original_videogames_df = pd.read_csv('Dataset/metacritic_18.07.2021_csv.csv')
# I cleaned the dataframe on colab
cleaned_videogames_df = pd.read_csv('Dataset/clean_dataset.csv')

st.header("Videogames")
st.write("The **project** is about  ................")

if st.sidebar.checkbox('show the original dataset'):
    st.write(original_videogames_df)
if st.sidebar.checkbox('show the dataset after the data cleaning'):
    st.write(cleaned_videogames_df)

#####################
# Show some interesting plot
#####################