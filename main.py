# Videogames Programming Project

import pandas as pd
import numpy as np
import seaborn as sns
import visualization_functions as vis
import streamlit as st
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

original_videogames_df = pd.read_csv('Dataset/metacritic_18.07.2021_csv.csv')
# I did the complete EDA part and cleaned the dataframe on colab notebook uploaded on github
cleaned_videogames_df = pd.read_csv('Dataset/clean_dataset.csv')

# Initializing web page
st.set_page_config(layout = 'centered')
st.header('Videogames: Metacritic Vs Userscore ratings')
st.markdown("**The original dataset could be found here:** [Metacritic dataset](https://www.kaggle.com/datasets/taranenkodaria/videogame-metacritic)")
st.markdown("""
            The dataset contains a series of videogames from the 1998 to 2021 with their rating given by a critic (Metacritic) and from the players.\n
            The features of this datasets are: 
            - **Titles**: The game names.
            - **Platforms**: The game platform, games can have an implementation on several platforms.
            - **Metascore**: The rating put down metacritic.com.
            - **Userscore**: The user rating, may not be available for new games.
            - **Genre**: The game genre, games can have more than one genre.
            - **Date**: The date of release of the game.
            """)


#####################
# some EDA
#####################
st.sidebar.write('Choose which dataset yuo want to see:')
if st.sidebar.checkbox('original dataset'):
        st.subheader('Original dataset')
        st.write(original_videogames_df)
        st.write('Numerical value before the cleaning:')
        st.write(original_videogames_df.describe().T)

if st.sidebar.checkbox('dataset after the data cleaning'):
        st.subheader('Dataset after the data cleaning')
        st.write(cleaned_videogames_df)
        st.write('Numerical value after the cleaning:')
        st.write(cleaned_videogames_df.describe().T)

#####################
# Visualization Section
#####################

st.write("## Visualizations")

selection = st.selectbox('Select if you want to see the **number of game per platform** or **number of game per genre**:', [" ","genre","platform"])
if selection == "genre":
        # number of games for each platforms
        general_counts = cleaned_videogames_df[selection].value_counts()
        vis.plot_bar((12, 8), general_counts, col='lightgreen', title='Number of Games per ' + selection, x_lab = selection, y_lab='Number of Games', rot=90)
elif selection == "platform":
        general_counts = cleaned_videogames_df[selection].value_counts()
        vis.plot_bar((12, 8), general_counts, col = 'lightblue', title = 'Number of Games per ' + selection, x_lab = selection, y_lab = 'Number of Games', rot=90)



# Comparison between Metascore and Userscore
st.write("### Comparison between Metascore and Userscore")

st.write("### Comparison of the two distribution")

col1, col2 = st.columns(2)
with col1:
        vis.plot_histogram(cleaned_videogames_df, 'metascore', 'Metascore Distribution', color='blue')
with col2:
        vis.plot_histogram(cleaned_videogames_df, 'userscore', 'Userscore Distribution', color='orange')

st.write("We can see that both the score metric follow a Normal Distribution.")


