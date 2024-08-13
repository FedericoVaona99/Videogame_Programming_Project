# Videogames Programming Project

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import visualization_functions as vis
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)

original_videogames_df = pd.read_csv('Dataset/metacritic_18.07.2021_csv.csv')
# I did the complete EDA part and cleaned the dataframe on colab notebook uploaded on github
cleaned_videogames_df = pd.read_csv('Dataset/clean_dataset.csv')

# Initializing web page
st.set_page_config(layout = 'wide')
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
st.sidebar.write('Choose which Dataset you want to see:')
if st.sidebar.checkbox('Original Dataset'):
        st.subheader('Original dataset')
        st.dataframe(original_videogames_df, width=1500)
        st.write('Numerical value before the cleaning:')
        st.write(original_videogames_df.describe().T)

if st.sidebar.checkbox('Cleaned Dataset'):
        st.subheader('Dataset after the data cleaning')
        st.dataframe(cleaned_videogames_df, width= 1500)
        st.write('Numerical value after the cleaning:')
        st.write(cleaned_videogames_df.describe().T)
else:
        st.write()
#####################
# Visualization Section
#####################


with st.expander("VISUALIZATION"):

        selection = st.selectbox('Select if you want to see stats for **platform** or **genre** of the videogame:', [" ","genre","platform"])
        if selection == "genre":
                # number of games for each platforms
                general_counts = cleaned_videogames_df[selection].value_counts()
                vis.plot_bar((12, 6), general_counts, col='lightgreen', title='Number of Games per ' + selection, x_lab = selection, y_lab='Number of Games', rot=90)

                # Average User Score by Genre
                avg_userscore = cleaned_videogames_df.groupby(selection)['userscore'].mean()
                vis.plot_bar(fig_size=(12, 6), datas=avg_userscore, col='lightgreen', title='Average User Score by '+ selection,
                             x_lab=selection, y_lab='Average User Score', rot=90)

                # Average Metacritic score by Genre
                avg_metascore = cleaned_videogames_df.groupby(selection)['metascore'].mean()
                vis.plot_bar(fig_size=(12, 6), datas=avg_metascore, col='lightgreen',
                             title='Average Metacritic Score by ' + selection,
                             x_lab=selection, y_lab='Average Metacritic Score', rot=90)

                # top 3 Best game for genre selected
                unique_genres = sorted(cleaned_videogames_df['genre'].unique())
                genre_selected = st.multiselect('Select for which genre you want to see the best game:', unique_genres)

                col1, col2 = st.columns(2)

                if genre_selected:
                        with col1:
                                vis.display_top_games_by_score_type(cleaned_videogames_df,'genre', genre_selected, 'metascore', 3)
                        with col2:
                                vis.display_top_games_by_score_type(cleaned_videogames_df,'genre', genre_selected, 'userscore', 3)
                else:
                        st.write("Please, select at least one genre to see the top games.")

        elif selection == "platform":

                general_counts = cleaned_videogames_df[selection].value_counts()
                vis.plot_bar((12, 6), general_counts, col = 'lightblue', title = 'Number of Games per ' + selection, x_lab = selection, y_lab = 'Number of Games', rot=90)

                # Average User Score by Platform
                avg_userscore = cleaned_videogames_df.groupby(selection)['userscore'].mean()
                vis.plot_bar(fig_size=(12, 6), datas=avg_userscore, col='lightblue', title='Average User Score by '+ selection,
                             x_lab=selection, y_lab='Average User Score', rot=90)

                # Average Metacritic score by Platform
                avg_metascore = cleaned_videogames_df.groupby(selection)['metascore'].mean()
                vis.plot_bar(fig_size=(12,6), datas=avg_metascore, col='lightblue',
                             title='Average Metacritic Score by ' + selection,
                             x_lab=selection, y_lab='Average Metacritic Score', rot=90)

                # top 3 Best game for platform selected
                unique_platforms = sorted(cleaned_videogames_df['platform'].unique())
                platforms_selected = st.multiselect('Select for which platform you want to see the best game:', unique_platforms)

                col1, col2 = st.columns(2)

                if platforms_selected:
                        with col1:
                                vis.display_top_games_by_score_type(cleaned_videogames_df, 'platform', platforms_selected, 'metascore', 3)
                        with col2:
                                vis.display_top_games_by_score_type(cleaned_videogames_df, 'platform', platforms_selected, 'userscore', 3)
                else:
                        st.write("Please, select at least one platform to see the top games.")





        # Comparison between Metascore and Userscore
        st.write("### Comparison between Metascore and Userscore")

        st.write("### Comparison of the two distribution")

        st.write("We can see that both the score metric follow a Normal Distribution.")

        col3, col4 = st.columns(2)
        with col3:
                vis.plot_histogram(cleaned_videogames_df, 'metascore', 'Metascore Distribution', color='blue')
        with col4:
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
        plt.figure(figsize=(12, 6))
        sns.histplot(score_diff_df['score_diff'], bins=20, color='purple')
        plt.title('Distribution of the differences between Metascore and Userscore rating')
        plt.xlabel('Difference (Metascore - Userscore)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(plt)

        st.write("We can see that the peak of the distribution is near 0, then the ratings often corresponds; but there are several cases where we have huge differences between the userscore and the metacritic scores with differences of 30-40 points")

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

        # Convert 'date' column to datetime if not already
        cleaned_videogames_df['date'] = pd.to_datetime(cleaned_videogames_df['date'])

        # Set the 'date' column as the index of the DataFrame
        cleaned_videogames_df.set_index('date', inplace=True)

        # Resample and calculate the mean of user scores and metascores by year
        yearly_avg_userscores = cleaned_videogames_df['userscore'].resample('Y').mean()
        yearly_avg_metascores = cleaned_videogames_df['metascore'].resample('Y').mean()

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_avg_userscores.index, yearly_avg_userscores.values, label='Average User Score', marker='o', color='blue')
        plt.plot(yearly_avg_metascores.index, yearly_avg_metascores.values, label='Average Metascore', marker='o', color='red')

        # Set major and minor ticks format
        ax = plt.gca()  # Get current axis
        ax.xaxis.set_major_locator(mdates.YearLocator())  # Set major locator to every year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format x-axis labels to show only the year
        # Rotate x-axis labels to prevent overlap
        ax.tick_params(axis='x', rotation=45)  # Rotate labels to 45 degrees

        plt.title('Comparison of Average User Scores and Metascores Over Time')
        plt.xlabel('Year')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        col5, col6 = st.columns(2)
        cleaned_videogames_df.reset_index(inplace=True)

        # Convert the 'date' column to datetime and extract the year
        cleaned_videogames_df['date'] = pd.to_datetime(cleaned_videogames_df['date'])
        cleaned_videogames_df['year'] = cleaned_videogames_df['date'].dt.year

        with col5:
                # Find the game with the highest metascore for each year
                best_games_per_year_meta = cleaned_videogames_df.loc[cleaned_videogames_df.groupby('year')['metascore'].idxmax()]
                best_games_per_year_meta.reset_index(inplace=True)

                # Ensure the year is displayed without decimals
                best_games_per_year_meta['year'] = best_games_per_year_meta['year'].astype(str)

                # Display the DataFrame in Streamlit
                st.write("Top game for each year per Metacritic")
                st.dataframe(best_games_per_year_meta[['title','platform','metascore','userscore','year']], width=750)

        with col6:
                # Find the game with the highest userscore for each year
                best_games_per_year_user = cleaned_videogames_df.loc[
                        cleaned_videogames_df.groupby('year')['userscore'].idxmax()]
                best_games_per_year_user.reset_index(inplace=True)

                best_games_per_year_user['year'] = best_games_per_year_user['year'].astype(str)
                st.write("Top game for each year per Players")
                st.dataframe(best_games_per_year_user[['title', 'platform', 'metascore', 'userscore', 'year']], width=750)


#######################
### MACHINE LEARNING
#######################

with st.expander("MACHINE LEARNING"):

        st.markdown("""
        In this part, i made 2 distinct classification model:
        - **Classification by Genre and Platform:** The model try to classify the videogames in one of these 3 categories -> [Good Game, Average Game, Bad Game] knowing only the genre and the platform of the videogame.
        - **Classification by Metascore:** The model try to classify the videogames in a **succesfull** game for the players or **not successful** knowing only the score obtained by metacritic.
        """)

        model_selection = st.selectbox('Select which model you want to display:', ["","Classification by Genre and Platform","Classification by Metascore"])

        if model_selection == "Classification by Genre and Platform":

                df = pd.read_csv('Dataset/clean_dataset.csv')
                X_final, y = vis.prepare_data(df, "type1")

                method_selected = st.selectbox("Select which classification method to use:",
                                               ["Logistic Regression", "Support Vector Machine",
                                                "Gradient Boosting", "K-Nearest Neighbors", "Random Forest"])
                options = [round(x * 0.05, 2) for x in range(1, 20)]
                test_size = st.select_slider('Slide to select the test size', options=options, value=0.2)
                X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_size, random_state=42)

                if method_selected == "Logistic Regression":
                        vis.loading_data("Logistic Regression")
                        model = LogisticRegression(random_state=42)
                elif method_selected == "Support Vector Machine":
                        vis.loading_data("Support Vector Machine")
                        model = SVC(random_state=42)
                elif method_selected == "Gradient Boosting":
                        vis.loading_data("Gradient Boosting")
                        model = GradientBoostingClassifier(random_state=42)
                elif method_selected == "K-Nearest Neighbors":
                        vis.loading_data("K-Nearest Neighbors")
                        model = KNeighborsClassifier()
                elif method_selected == "Random Forest":
                        vis.loading_data("Random Forest")
                        model = RandomForestClassifier(random_state=42)

                # Evaluation
                accuracy, precision, recall, f1 = vis.train_and_evaluate_model(X_train, X_test, y_train, y_test, model, "type1")

                # Display results
                st.write(f"Accuracy: {accuracy * 100} %")
                st.write(f"Precision: {precision * 100} %")
                st.write(f"Recall: {recall * 100} %")
                st.write(f"F1 Score: {f1 * 100} %")

                vis.plot_Classification_results(accuracy, precision, recall, f1, method_selected)

        elif model_selection == "Classification by Metascore":

                df = pd.read_csv('Dataset/clean_dataset.csv')
                X_final, y = vis.prepare_data(df, "type2")


                method_selected = st.selectbox("Select which classification method to use:",
                                               ["Logistic Regression", "Support Vector Machine",
                                                "Gradient Boosting", "K-Nearest Neighbors", "Random Forest"])

                options = [round(x * 0.05, 2) for x in range(1, 20)]
                test_size = st.select_slider('Slide to select the test size', options=options, value=0.2)
                X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size= test_size, random_state=42)

                if method_selected == "Logistic Regression":
                        vis.loading_data("Logistic Regression")
                        model = LogisticRegression(random_state=42)
                elif method_selected == "Support Vector Machine":
                        vis.loading_data("Support Vector Machine")
                        model = SVC(random_state=42)
                elif method_selected == "Gradient Boosting":
                        vis.loading_data("Gradient Boosting")
                        model = GradientBoostingClassifier(random_state=42)
                elif method_selected == "K-Nearest Neighbors":
                        vis.loading_data("K-Nearest Neighbors")
                        model = KNeighborsClassifier()
                elif method_selected == "Random Forest":
                        vis.loading_data("Random Forest")
                        model = RandomForestClassifier(random_state=42)

                # Evaluation
                accuracy, precision, recall, f1 = vis.train_and_evaluate_model(X_train, X_test, y_train, y_test, model, "type2")

                # Display results
                st.write(f"Accuracy: {accuracy * 100} %")
                st.write(f"Precision: {precision * 100} %")
                st.write(f"Recall: {recall * 100} %")
                st.write(f"F1 Score: {f1 * 100} %")

                vis.plot_Classification_results(accuracy, precision, recall, f1, method_selected)
        else:
                st.write()