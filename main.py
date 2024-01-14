# Videogames Programming Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

' Data Exploration '

videogames_df = pd.read_csv('Dataset/metacritic_18.07.2021_csv.csv')
' Cleaning up the Dataset '

# Convert date to datetime object
videogames_df['date'] = pd.to_datetime(videogames_df['date'])

# I create the boolean mask to find all the 'tbd'
to_be_decided_mask = videogames_df.userscore == 'tbd'

' Replacing "tbd" and the NaN values. '

# First, I turn 'tbd' values into NaN values
videogames_df.loc[to_be_decided_mask,'userscore'] = np.nan

# Now I'm converting the "userscore" column to the float type.
videogames_df.userscore = videogames_df.userscore.astype(float)

# Now I'm replacing all the null values with the mean value.
videogames_df.userscore.fillna(videogames_df.userscore.mean(), inplace = True)

# I want to display a maximum of 2 decimals for the userscore.
videogames_df.userscore = videogames_df.userscore.round(2)

# I'm also replacing the NaN values in the metascore column with the average rating
videogames_df.metascore.fillna(videogames_df.metascore.mean(), inplace = True)
videogames_df.metascore = videogames_df.metascore.round(1)

# Now, to make the userscore comparable to the metascore in the future analysis, I'll scale it to a 100-point scale.
videogames_df.userscore = videogames_df.userscore * 10

# Cleaning platforms name iOS/n...(Apple Arcade) to iOS (Apple Arcade)
videogames_df.loc[videogames_df['platforms'] == 'iOS\n                                                                                    \xa0(Apple Arcade)','platforms'] = 'iOS (Apple Arcade)'
videogames_df['platforms'].unique()


' Show some interesting plot'