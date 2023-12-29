# Videogames Programming Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

' Data Exploration '

videogames_df = pd.read_csv('Dataset/metacritic_18.07.2021_csv.csv')
print(videogames_df.head(10))
print(videogames_df.tail())

videogames_df.info()                # shows if there are null values

print(videogames_df.describe().T)
print('\nThe shape of the dataframe is: ', videogames_df.shape)

' Cleaning up the Dataset '

# Convert date to datetime object
videogames_df['date'] = pd.to_datetime(videogames_df['date'])

# I create the boolean mask to find all the 'tbd'
to_be_decided_mask = videogames_df.userscore == 'tbd'
print(videogames_df[to_be_decided_mask])

# Before to lose track of the tbd values, I want to see how many games don't have still an evaluation score
user_tbd_count = videogames_df['userscore'][to_be_decided_mask].count()
print('Number of userscore with tbd value = ', user_tbd_count)

# I print also the null values to see how many values of the feature are missing
user_null_count = videogames_df.userscore.isnull().sum()
print('NaN values for the userscore are instead = ', user_null_count)

print('Then, are missing', user_null_count + user_tbd_count, 'values for the userscore feature')

# I see that there are still games from the 2000s that do not have a rating, so the "to be determined" value is not only for more recent games.
print(videogames_df[to_be_decided_mask].sort_values(by='date').head())

' Replacing "tbd" and the NaN values. '

# First, I turn 'tbd' values into NaN values
videogames_df.loc[to_be_decided_mask,'userscore'] = np.nan

# Now I'm converting the "userscore" column to the float type.
videogames_df.userscore = videogames_df.userscore.astype(float)

# Now I'm replacing all the null values with the mean value.
videogames_df.userscore.fillna(videogames_df.userscore.mean(), inplace = True)

# I want to display a maximum of 2 decimals for the userscore.
videogames_df.userscore = videogames_df.userscore.round(2)
print(videogames_df.userscore.head())

# I'm also replacing the NaN values in the metascore column with the average rating
print(videogames_df.metascore.tail())
videogames_df.metascore.fillna(videogames_df.metascore.mean(), inplace = True)
videogames_df.metascore = videogames_df.metascore.round(1)
print(videogames_df.metascore.tail())

# I'm checking to ensure that there are no remaining null values.
print(videogames_df.isnull().sum())

# Now, to make the userscore comparable to the metascore in the future analysis, I'll scale it to a 100-point scale.
videogames_df.userscore = videogames_df.userscore * 10
print(videogames_df.userscore.head())

videogames_df.info()
print(videogames_df.describe().T)

# Checking unique value of genre
videogames_df['genre'].unique()

# Checking unique value of platforms
videogames_df['platforms'].unique()

# Cleaning platforms name iOS/n...(Apple Arcade) to iOS (Apple Arcade)
videogames_df.loc[videogames_df['platforms'] == 'iOS\n                                                                                    \xa0(Apple Arcade)','platforms'] = 'iOS (Apple Arcade)'
videogames_df['platforms'].unique()


' Show some interesting plot'