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
# I clean the dataframe in a copy, to keep the original dataframe and show it in the web page
cleaned_videogames_df = original_videogames_df.copy()

#####################
# Cleaning up the Dataset
#####################

# Convert date to datetime object
cleaned_videogames_df['date'] = pd.to_datetime(cleaned_videogames_df['date'])

# I create the boolean mask to find all the 'tbd'
to_be_decided_mask = cleaned_videogames_df.userscore == 'tbd'

#####   Replacing "tbd" and the NaN values.

# First, I turn 'tbd' values into NaN values
cleaned_videogames_df.loc[to_be_decided_mask,'userscore'] = np.nan

# Now I'm converting the "userscore" column to the float type.
cleaned_videogames_df.userscore = cleaned_videogames_df.userscore.astype(float)

# Now I'm replacing all the null values with the mean value.
cleaned_videogames_df.userscore.fillna(cleaned_videogames_df.userscore.mean(), inplace = True)

# I want to display a maximum of 2 decimals for the userscore.
cleaned_videogames_df.userscore = cleaned_videogames_df.userscore.round(2)

# I'm also replacing the NaN values in the metascore column with the average rating
cleaned_videogames_df.metascore.fillna(cleaned_videogames_df.metascore.mean(), inplace = True)
cleaned_videogames_df.metascore = cleaned_videogames_df.metascore.round(1)

# Now, to make the userscore comparable to the metascore in the future analysis, I'll scale it to a 100-point scale.
cleaned_videogames_df.userscore = cleaned_videogames_df.userscore * 10

# Cleaning platforms name iOS/n...(Apple Arcade) to iOS (Apple Arcade)
cleaned_videogames_df.loc[cleaned_videogames_df['platforms'] == 'iOS\n                                                                                    \xa0(Apple Arcade)','platforms'] = 'iOS (Apple Arcade)'

st.title("Videogames")
st.write("The project is .......")

if st.sidebar.checkbox('show the dataframe before the data cleaning'):
    st.write(original_videogames_df.head())
if st.sidebar.checkbox('show the dataframe after the data cleaning'):
    st.write(cleaned_videogames_df.head())

#####################
# Show some interesting plot
#####################