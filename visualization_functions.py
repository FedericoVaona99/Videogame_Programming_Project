import streamlit as st
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

def plot_boxplot(xlabel, ylabel, data,title, palette='Blues' ):
    sns.boxplot(x=xlabel, y=ylabel, data=data, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    st.pyplot(plt)

