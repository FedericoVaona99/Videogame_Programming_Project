import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to plot a bar chart
def plot_bar(fig_size, datas, col = 'blue', title = '', x_lab = '', y_lab = '', rot=0 ):
  plt.figure(figsize = fig_size)
  bars = plt.bar(datas.index, datas.values, color=col)
  plt.title(title)
  plt.xlabel(x_lab)
  plt.ylabel(y_lab)
  plt.xticks(rotation= rot)
  # Annotazione dei valori sopra le barre
  for bar in bars:
      yval = bar.get_height()
      plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)
  st.pyplot()

def plot_histogram(data, column, title, bins=50, color='blue'):
    fig, ax = plt.subplots()
    sns.histplot(data[column].dropna(), bins=bins, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(column)
    st.pyplot(fig)

def plot_correlation_heatmap(data, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = data.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Function to get top 5 games by Metascore for each selected genre
def display_top_games_by_score_type(df,feature, features, score_type, num_games=3):
    all_top_games = []

    # Loop over each selected genre and append the top games to the list
    for feat in features:
        # Filter the DataFrame for the genre
        genre_df = df[df[feature] == feat]
        # Sort by the specified score type and take the top entries
        top_games = genre_df.nlargest(num_games, score_type)[['title', feature, score_type]]
        all_top_games.append(top_games)

    # Concatenate all DataFrames in the list into a single DataFrame and reset the index
    combined_top_games = pd.concat(all_top_games).reset_index(drop=True)

    # Display the concatenated DataFrame in Streamlit
    st.dataframe(combined_top_games, width=1500, height=300)

