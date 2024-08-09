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

selection = st.selectbox('Select if you want to see stats for **platform** or **genre** of the videogame:', [" ","genre","platform"])
if selection == "genre":
        # number of games for each platforms
        general_counts = cleaned_videogames_df[selection].value_counts()
        vis.plot_bar((12, 8), general_counts, col='lightgreen', title='Number of Games per ' + selection, x_lab = selection, y_lab='Number of Games', rot=90)

        # Average User Score by Genre
        avg_userscore = cleaned_videogames_df.groupby(selection)['userscore'].mean()
        vis.plot_bar(fig_size=(14, 6), datas=avg_userscore, col='lightgreen', title='Average User Score by '+ selection,
                     x_lab=selection, y_lab='Average User Score', rot=90)

        # Average Metacritic score by Genre
        avg_metascore = cleaned_videogames_df.groupby(selection)['metascore'].mean()
        vis.plot_bar(fig_size=(14, 6), datas=avg_metascore, col='lightgreen',
                     title='Average Metacritic Score by ' + selection,
                     x_lab=selection, y_lab='Average Metacritic Score', rot=90)

elif selection == "platform":

        general_counts = cleaned_videogames_df[selection].value_counts()
        vis.plot_bar((12, 8), general_counts, col = 'lightblue', title = 'Number of Games per ' + selection, x_lab = selection, y_lab = 'Number of Games', rot=90)

        # Average User Score by Platform
        avg_userscore = cleaned_videogames_df.groupby(selection)['userscore'].mean()
        vis.plot_bar(fig_size=(14, 6), datas=avg_userscore, col='lightblue', title='Average User Score by '+ selection,
                     x_lab=selection, y_lab='Average User Score', rot=90)

        # Average Metacritic score by Platform
        avg_metascore = cleaned_videogames_df.groupby(selection)['metascore'].mean()
        vis.plot_bar(fig_size=(14, 6), datas=avg_metascore, col='lightblue',
                     title='Average Metacritic Score by ' + selection,
                     x_lab=selection, y_lab='Average Metacritic Score', rot=90)





# Comparison between Metascore and Userscore
st.write("### Comparison between Metascore and Userscore")

st.write("### Comparison of the two distribution")

st.write("We can see that both the score metric follow a Normal Distribution.")

col1, col2 = st.columns(2)
with col1:
        vis.plot_histogram(cleaned_videogames_df, 'metascore', 'Metascore Distribution', color='blue')
with col2:
        vis.plot_histogram(cleaned_videogames_df, 'userscore', 'Userscore Distribution', color='orange')



plt.figure(figsize=(14, 6))

# Metascore's distrib
sns.histplot(cleaned_videogames_df['metascore'], bins=100, color='red', label='Metascore', alpha=0.8)

# Userscore distrib
sns.histplot(cleaned_videogames_df['userscore'], bins=100, color='yellow', label='Userscore', alpha=0.5)

# Titles and labels
plt.title('The two distribution together')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.legend()
st.pyplot(plt)
st.write("Metacritic scores tend to be higher ")

# Calcolo delle differenze tra metascore e userscore
# Before, I remove the outlier introduced during the data cleaning
score_diff_df = original_videogames_df.copy()
to_be_decided_mask = score_diff_df.userscore == 'tbd'
score_diff_df.loc[to_be_decided_mask,'userscore'] = np.nan
score_diff_df.userscore = score_diff_df.userscore.astype(float)
score_diff_df.userscore = score_diff_df.userscore * 10
score_diff_df.userscore = score_diff_df.userscore.round()
score_diff_df.dropna(subset=['metascore', 'userscore'], inplace=True)

#Then
score_diff_df['score_diff'] = score_diff_df['metascore'] - score_diff_df['userscore']

# Distribuzione delle differenze
plt.figure(figsize=(7, 6))
sns.histplot(score_diff_df['score_diff'], bins=20, color='purple')
plt.title('Distribution of the differences between Metascore and Userscore rating')
plt.xlabel('Difference (Metascore - Userscore)')
plt.ylabel('Frequency')

plt.tight_layout()
st.pyplot(plt)

st.write("We can see that the peak of the distribution is nar 0, then the ratings often corresponds; but there are several cases where we have huge differences between the userscore and the metacritic scores with differences of 30-40 points")

# Correlation Heatmap
st.write("### Correlation Heatmap without the categorical variables")
vis.plot_correlation_heatmap(cleaned_videogames_df, 'Correlation between Metascore and Userscore')

st.write("### Correlation Heatmap with the categorical variables")

# Identify the top 5 genres and platforms based on occurrence
top_genres = cleaned_videogames_df['genre'].value_counts().head(5).index.tolist()
top_platforms = cleaned_videogames_df['platform'].value_counts().head(5).index.tolist()

# Filter the DataFrame to include only rows with top genres and platforms
filtered_df = cleaned_videogames_df[
cleaned_videogames_df['genre'].isin(top_genres) &
cleaned_videogames_df['platform'].isin(top_platforms)
]

# One-Hot Encode 'genre' and 'platform' for the filtered DataFrame
genre_encoded = pd.get_dummies(filtered_df['genre'], prefix='g')
platform_encoded = pd.get_dummies(filtered_df['platform'], prefix='p')

# Combine the encoded columns with the original DataFrame
encoded_df = pd.concat([filtered_df[['metascore', 'userscore']], genre_encoded, platform_encoded], axis=1)

# Compute the correlation matrix for all variables
correlation_matrix = encoded_df.corr()

# Plotting the complete correlation matrix
plt.figure(figsize=(len(correlation_matrix.columns), len(correlation_matrix.columns) * 0.5))  # Size the figure to fit all variables
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Complete Correlation Matrix with Top 5 Genres and Platforms')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)   # Ensure y-axis labels are horizontal for better readability
plt.tight_layout()  # Adjust layout to not cut off labels

# Use Streamlit's function to show the plot
st.pyplot(plt)


