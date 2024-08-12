import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    # Step 1: Adjust aggregation to group by both 'title' and 'genre'
    aggregated_df = df.groupby(['title', feature]).agg({score_type: 'mean'}).reset_index()

    # Step 2: Loop over each selected feature value and append the top games to the list
    for feat in features:
        # Filter the DataFrame for the specific feature
        filtered_df = aggregated_df[aggregated_df[feature] == feat]
        # Sort by the specified score type and take the top entries
        top_games = filtered_df.nlargest(num_games, score_type)[['title', feature, score_type]]
        all_top_games.append(top_games)

    # Concatenate all DataFrames in the list into a single DataFrame and reset the index
    combined_top_games = pd.concat(all_top_games).reset_index(drop=True)

    # Display the concatenated DataFrame in Streamlit
    st.dataframe(combined_top_games, width=1500, height=300)

# Funzione per simulare il caricamento
def loading_data(section):

    # Mostra uno spinner durante il caricamento
    with st.spinner(f"Caricamento della sezione {section}..."):
        time.sleep(2)  # Simula un'operazione di caricamento
    st.success(f"{section} evaluation completed!")


###########################
## Machine Learning function
###########################

# Function to load and prepare data
def prepare_data(df, model_type):

    if model_type == "Regression":
        X = df[['metascore', 'genre', 'platform']]
        y = df['userscore']
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X[['genre', 'platform']]).toarray()
        feature_names = encoder.get_feature_names_out(['genre', 'platform'])
        X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)
        X_final = pd.concat([df[['metascore']], X_encoded_df], axis=1)
    elif model_type == "Classification":
        success_threshold = 75
        df['success'] = np.where(df['userscore'] >= success_threshold, 1, 0)
        X_final = df[['metascore']]
        y = df['success']
    return X_final, y

# Function to train and evaluate a model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model,type):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if type=="Regression":
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return mse, r2, mae, predictions

    elif type=="Classification":
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary', zero_division=0)
        recall = recall_score(y_test, predictions, average='binary')
        f1 = f1_score(y_test, predictions, average='binary')
        return accuracy, precision, recall, f1

# Function to plot results
def plot_Regression_results(y_test, predictions, plot_type):
    if plot_type == "Regression":
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Comparison of Actual vs. Predicted Values')
        st.pyplot(plt)

def plot_Classification_results(accuracy, precision, recall, f1,method_selected):
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Plotting
    results_df = pd.DataFrame([results])  # Convert dictionary to DataFrame
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