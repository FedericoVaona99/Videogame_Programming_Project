import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Disable the deprecation warning for Pyplot in Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to plot a bar chart
def plot_bar(fig_size, datas, col = 'blue', title = '', x_lab = '', y_lab = '', rot=0 ):

  plt.figure(figsize = fig_size)
  bars = plt.bar(datas.index, datas.values, color=col)
  plt.title(title)
  plt.xlabel(x_lab)
  plt.ylabel(y_lab)
  plt.xticks(rotation= rot)

  # Annotate values above the bars
  for bar in bars:
      yval = bar.get_height()
      plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=10)
  st.pyplot()

# Function to plot two histograms with the same y-axis scale
def plot_histograms_with_same_scale(data, col1, col2, title1, title2, bins=50, color1='blue', color2='orange'):

    # Create figures and axes for the two plots with shared y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot the first histogram (Metascore)
    sns.histplot(data[col1].dropna(), bins=bins, ax=ax1, color=color1)
    ax1.set_title(title1)
    ax1.set_xlabel(col1)

    # Plot the second histogram (Userscore)
    sns.histplot(data[col2].dropna(), bins=bins, ax=ax2, color=color2)
    ax2.set_title(title2)
    ax2.set_xlabel(col2)

    # # Sync the y-axis limits between both plots
    max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, max_y)
    ax2.set_ylim(0, max_y)

    plt.tight_layout()
    st.pyplot(fig)

# Function to plot a single histogram
def plot_histogram(data, column, title, bins=50, color='blue'):

    fig, ax = plt.subplots()
    sns.histplot(data[column], bins=bins, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(column)
    st.pyplot(fig)

# Function to plot a correlation heatmap
def plot_correlation_heatmap(data):

    fig, ax = plt.subplots(figsize=(4, 3))
    corr_matrix = data.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=2)
    plt.tight_layout()
    st.pyplot(fig)

# Function to get top 3 games by score for genre or platform
def display_top_games_by_score_type(df,column, features_selected, score_type, num_games=3):

    all_top_games = [] # store the top games for each feature

    # Group the data by 'title' and the chosen feature (genre or platform) and calculate the average score (for metascore or userscore)
    aggregated_df = df.groupby(['title', column]).agg({score_type: 'mean'}).reset_index()

    # Loop over each feature value and append the top games to the list
    for feat in features_selected:

        # Filter the df to only include rows where the feature matches the current feature value
        filtered_df = aggregated_df[aggregated_df[column] == feat]

        # Sort the data by the score and get the top 3 games
        top_games = filtered_df.nlargest(num_games, score_type)[['title', column, score_type]]
        # Add the top games for this feature to the list
        all_top_games.append(top_games)

    # Combine all the dataFrames of "all_top_games" into a single df
    combined_top_games = pd.concat(all_top_games).reset_index(drop=True)

    # Display the combined df in Streamlit
    st.dataframe(combined_top_games, width=1500, height=300)

# Function to simulate a loading spinner or progress bar
def loading_data(section, type = "type1"):

    if type == "type2":
        # Shows a spinner in the loading phase
        with st.spinner(f"Loading of the {section} method..."):
            time.sleep(2)  # Simulate a loading operation
        st.success(f"{section} evaluation completed!")

    elif type == "type1":
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.04)  # Simulate a loading operation
            progress_bar.progress(percent_complete + 1)
        st.success(f"{section} evaluation completed!")


###########################
## Machine Learning function
###########################

# Function to load and prepare data for classification
def prepare_data(df, classification_type, features = []):

    # Quality Classification
    if classification_type == "type1":

        # Define categories based on metascore (Bad, Average, Good)
        bins = [0, 65, 80, 100]
        labels = ['Bad Game', 'Average Game', 'Good Game']
        df['category'] = pd.cut(df['metascore'], bins=bins, labels=labels, include_lowest=True)

        # Prepare predictive variables using selected features
        X_final = df[features]

        # Apply One-Hot Encoding if features include 'genre' or 'platform'
        if 'genre' in features or 'platform' in features:
            X_final = pd.get_dummies(X_final, columns=[col for col in ['genre', 'platform'] if col in features])

        y = df['category'] # Target variable

    # Game Success Classification
    elif classification_type == "type2":

        # Define success threshold for userscore
        success_threshold = 75
        df['success'] = np.where(df['userscore'] >= success_threshold, 1, 0) # Binary classification: success or not

        X_final = df[['metascore']] # Use Metascore as the only predictive variable
        y = df['success'] # Target variable

    return X_final, y

# Function to train and evaluate a model based on the classification type (only 'macro' / 'binary' change)
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model,type):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Quality Classification
    if type =="type1":
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro', zero_division=0)
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')

    # Game Success Classification
    elif type =="type2":
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary', zero_division=0)
        recall = recall_score(y_test, predictions, average='binary')
        f1 = f1_score(y_test, predictions, average='binary')

    return accuracy, precision, recall, f1

# Function to plot the classification results
def plot_Classification_results(accuracy, precision, recall, f1, method_selected):

    # Store the evaluation metrics in a dictionary
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Convert the dictionary to a DataFrame and plotting
    results_df = pd.DataFrame([results])
    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.plot(kind='bar', ax=ax, colormap='viridis')
    plt.title(f'{method_selected} Performance Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.xticks(ticks=[0], labels=[method_selected], rotation=0)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')

    plt.tight_layout()
    st.pyplot(fig)