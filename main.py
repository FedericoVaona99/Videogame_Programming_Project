# Videogames Programming Project

# python -m streamlit run main.py

import pandas as pd
import numpy as np
import seaborn as sns

import visualization_functions as vis
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Disable the deprecation warning for Pyplot in Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

 # LOADING DATASETS (Original and cleaned)
 # I did the complete EDA part and cleaned the dataframe with comments on colab notebook uploaded on github to mantain the streamlit code more ordered

original_videogames_df = pd.read_csv('Dataset/metacritic_18.07.2021_csv.csv')
cleaned_videogames_df = pd.read_csv('Dataset/clean_dataset.csv')

# Initializing web page
st.set_page_config(layout = 'wide')
st.title('🎮 Videogames: Metacritic Vs Userscore Ratings')

st.markdown("""
**Explore the relationship between critic and player ratings across various genres, platforms, and time periods.**

This project provides insights into video game ratings by comparing critic scores from **Metacritic** with player scores. 
The dataset covers video games released between **1998 and 2021**.

**The original dataset can be found here:** [Metacritic dataset](https://www.kaggle.com/datasets/taranenkodaria/videogame-metacritic)

### 📊 Dataset Features:
- **Titles**: The names of the video games.
- **Platforms**: The platforms on which the games are available (e.g., PC, PlayStation, Xbox). Some games are available on multiple platforms.
- **Metascore**: The aggregated critic score from [Metacritic](https://www.metacritic.com), on a scale of 0 to 100.
- **Userscore**: The player score, provided by users on Metacritic, on a scale of 0 to 10. For some new or less popular games, the score may still be marked as 'tbd' (to be determined).
- **Genre**: The genre of the game (e.g., Action, RPG, Adventure). Some games may belong to multiple genres.
- **Date**: The official release date of the game.
---
""")


#####################
# part of the EDA
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


#######################
# VISUALIZATION SECTION
#######################


with st.expander("**VISUALIZATIONS**"):

        selection = st.selectbox('Select if you want to see stats for **platform** or **genre** of the videogame:', [" ","genre","platform"])

        if selection == "genre":
                # Show the number of games for each platforms
                general_counts = cleaned_videogames_df[selection].value_counts()
                vis.plot_bar((12, 6), general_counts, col='lightgreen', title='Number of Games per ' + selection, x_lab = selection, y_lab='Number of Games', rot=90)
                st.markdown("""
                **Observation**: The 'Action' genre dominates the dataset with a significantly higher number of games compared to other genres, such as 'Role-playing' and 'Adventure'. This suggests that action games are much more common in the market. 

                It's important to note that many games belong to multiple genres, and 'Action' is frequently one of them, further explaining its dominance. Other genres like 'Puzzle', 'Fighting', and 'Party' have considerably fewer games, indicating their niche appeal or smaller market presence.
                """)

                # Show Average User Score by Genre
                avg_userscore = cleaned_videogames_df.groupby(selection)['userscore'].mean()
                vis.plot_bar(fig_size=(12, 6), datas=avg_userscore, col='lightgreen', title='Average User Score by '+ selection,
                             x_lab=selection, y_lab='Average User Score', rot=90)
                st.markdown("""
                **Observation**: The average user scores across different genres are relatively similar, with all the genres receiving an average score between **69 and 75**. 

                Genres such as **'Fighting'** and **'Party'** tend to have slightly higher user ratings, indicating stronger engagement from players. On the other hand, **'Simulation'** has the lowest average score, suggesting that it may appeal less to the average player.

                Overall, there doesn't seem to be a huge variation in user scores between genres, which indicates a generally consistent perception from players regardless of the type of game.
                """)

                # Show Average Metacritic score by Genre
                avg_metascore = cleaned_videogames_df.groupby(selection)['metascore'].mean()
                vis.plot_bar(fig_size=(12, 6), datas=avg_metascore, col='lightgreen',
                             title='Average Metacritic Score by ' + selection,
                             x_lab=selection, y_lab='Average Metacritic Score', rot=90)
                st.markdown("""
                Comparing the average Metacritic scores with the average User scores by genre, we notice that both critics and users exhibit consistent average scores across genres, with ranges being relatively narrow.

                **'Fighting'** and **'Party'** genres are the higher average scores from users, while the **'Party'** genre for metacritic has the lowest average score and they have not a clearly favorred game.
                
                Overall, both critics and users seem to agree on the general ranking of genres, with only minor differences in the actual scores. This suggests that the perception of game quality between critics and players is fairly aligned across genres.
                """)

                # Show top 3 Best game for genre selected
                unique_genres = sorted(cleaned_videogames_df['genre'].unique())
                genre_selected = st.multiselect('Select one or more **genres** to view the top3-rated games:', unique_genres)

                col1, col2 = st.columns(2)

                if genre_selected:
                        with col1:
                                vis.display_top_games_by_score_type(cleaned_videogames_df,'genre', genre_selected, 'metascore', 3)
                        with col2:
                                vis.display_top_games_by_score_type(cleaned_videogames_df,'genre', genre_selected, 'userscore', 3)
                else:
                        st.write("Please, select at least one genre to display the top games.")

        elif selection == "platform":

                # Show the number of games per platform
                general_counts = cleaned_videogames_df[selection].value_counts()
                vis.plot_bar((12, 6), general_counts, col = 'lightblue', title = 'Number of Games per ' + selection, x_lab = selection, y_lab = 'Number of Games', rot=90)
                st.markdown("""
                **Observation**: The 'PC' platform has by far the highest number of games, with over **7,600 titles**, significantly more than any other platform. This likely reflects the openness of the PC platform for game development and the large number of indie and smaller-scale games available on PC.

                The next most popular platforms is **iOS**, showing that mobile gaming (iOS) remain important in the gaming landscape due to its accessibility and the widespread availability of mobile devices.

                Platforms such as **PlayStation 5** and **Xbox Series X** have very few games in comparison, due to their more recent releases.
                """)

                # Show Average User Score by Platform
                avg_userscore = cleaned_videogames_df.groupby(selection)['userscore'].mean()
                vis.plot_bar(fig_size=(12, 6), datas=avg_userscore, col='lightblue', title='Average User Score by '+ selection,
                             x_lab=selection, y_lab='Average User Score', rot=90)
                st.markdown("""
                **Observation**: The platform with the highest average user score is **Nintendo 64** with a score of **79**, followed closely by **PlayStation** at **77**. This suggests that games on these older platforms are generally highly rated by players. 

                On the other hand, **PlayStation 4** has the lowest average user score at **67**, indicating that games on this platform may not be as well-received by players.
                """)

                # Show Average Metacritic score by Platform
                avg_metascore = cleaned_videogames_df.groupby(selection)['metascore'].mean()
                vis.plot_bar(fig_size=(12,6), datas=avg_metascore, col='lightblue',
                             title='Average Metacritic Score by ' + selection,
                             x_lab=selection, y_lab='Average Metacritic Score', rot=90)
                st.markdown("""
                Comparing the average user scores and the average Metacritic scores by platform, we notice that for both users and critics, the **Nintendo 64** holds the top spot with the highest average scores (79 for both groups). This suggests a strong consensus between critics and players regarding the quality of games on this platform.

                For the lowest average scores, there is no consensus between critics and users. Players rate the **PlayStation 4** the lowest, with an average score of 67, while for critics, the **Wii** and **DS** have the lowest average score of 68.

                **General trends**:
                The range of scores is narrow for both groups, with most platforms receiving relatively positive scores. However, there seems to be a slight tendency for users to rate older platforms (like Dreamcast) higher than critics, while critics tend to give newer platforms (like PlayStation 4 and 5) marginally better ratings.
                
                ---
                """)

                # Show top 3 Best game for platform selected
                unique_platforms = sorted(cleaned_videogames_df['platform'].unique())
                platforms_selected = st.multiselect('Select one or more **platforms** to view the top3-rated games:', unique_platforms)

                col1, col2 = st.columns(2)

                if platforms_selected:
                        with col1:
                                vis.display_top_games_by_score_type(cleaned_videogames_df, 'platform', platforms_selected, 'metascore', 3)
                        with col2:
                                vis.display_top_games_by_score_type(cleaned_videogames_df, 'platform', platforms_selected, 'userscore', 3)
                else:
                        st.write("Please, select at least one platform to display the top games.")

        st.markdown("""
        ---
        """)


        # Comparison between Metascore and Userscore
        st.header("Comparison between Metascore and Userscore")

        st.write("##### We can see that both the score metric follow a Normal Distribution.")

        vis.plot_histograms_with_same_scale(
                cleaned_videogames_df,
                'metascore', 'userscore',
                'Metascore Distribution', 'Userscore Distribution',
                color1='blue', color2='orange'
        )

        st.markdown("""
                **Observations**: Both the Metascore and Userscore distributions follow a roughly normal (bell-shaped) distribution, although there are some key differences between the two:

                **Metascore Distribution**: The Metascore distribution is slightly skewed to the right, with the majority of scores falling between **70 and 85**. This indicates that critics tend to give relatively high scores to games, with less games receiving scores below 60 or over 85. The peak is centered around **72-73**, showing that most games are rated in the upper-middle range.

                **Userscore Distribution**: The Userscore distribution instead is slightly skewed to the left, with most scores between **65** and **80**, and a peak around **70-71**. Then, Userscore distribution shows that players are generally more critical or conservative in their ratings compared to critics, who tend to give higher scores more often.
                """)



        # Plot Meta and User scores distributions together for comparison
        plt.figure(figsize=(14, 6))

        # Metascore's distribution
        sns.histplot(cleaned_videogames_df['metascore'], bins=100, color='red', label='Metascore', alpha=0.8)

        # Userscore distribution
        sns.histplot(cleaned_videogames_df['userscore'], bins=100, color='yellow', label='Userscore', alpha=0.5)

        # Titles and labels
        plt.title('The two distribution together')
        plt.xlabel('Score')
        plt.ylabel('Frequency')

        plt.legend()
        st.pyplot(plt)
        st.markdown("""
                **Comparison**: Although both distributions have a similar general shape, the Metascore is more concentrated in the higher ranges, while the Userscore is skewed towards the mid-to-lower ranges. This suggests that critics tend to be more generous in their evaluations, whereas players are more critical and maybe they also could have higher expectations from game developers.
                """)

        # Computation of differences between metascore and userscore

        # Before, I remove the outlier introduced during the data cleaning ----> On Colab, there is the plot with the outliers
        score_diff_df = original_videogames_df.copy()
        to_be_decided_mask = score_diff_df.userscore == 'tbd'
        score_diff_df.loc[to_be_decided_mask,'userscore'] = np.nan
        score_diff_df.userscore = score_diff_df.userscore.astype(float)
        score_diff_df.userscore = score_diff_df.userscore * 10
        score_diff_df.userscore = score_diff_df.userscore.round()
        score_diff_df.dropna(subset=['metascore', 'userscore'], inplace=True)

        # After, I had at this dataframe with no outlier a new column ('score_diff')
        score_diff_df['score_diff'] = score_diff_df['metascore'] - score_diff_df['userscore']

        # Plot the score_diff distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(score_diff_df['score_diff'], bins=20, color='purple')
        plt.title('Distribution of the differences between Metascore and Userscore rating')
        plt.xlabel('Difference (Metascore - Userscore)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("""
                **Observation:** We can see that the peak of the distribution is near 0, then the ratings often corresponds; but there are several cases where we have huge differences between the userscore and the metacritic scores with differences of 30-40 points.
                """)


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
        ax.tick_params(axis='x', rotation=45)

        plt.title('Comparison of Average User Scores and Metascores Over Time')
        plt.xlabel('Year')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        st.markdown("""
        **Observation**: Both the average User Score and Metascore show a significant decline after 2000. Prior to this, both scores were consistently higher, hovering around 80-85.\n
        After 2001, the scores stabilized at a lower range, with User score remaining higher than Metascore until 2010, when the trend reversed.
        """)
        st.write("\n")

        col5, col6 = st.columns(2)
        cleaned_videogames_df.reset_index(inplace=True)

        #  and extract the year
        cleaned_videogames_df['year'] = cleaned_videogames_df['date'].dt.year

        with col5:

                # Group by year and find the index of the highest metascore for each year
                idx = cleaned_videogames_df.groupby('year')['metascore'].idxmax()
                # Select the rows with the highest metascore for each year
                best_games_per_year_meta = cleaned_videogames_df.loc[idx]
                # Reset the index for the final output
                best_games_per_year_meta.reset_index(drop=True, inplace=True)

                # Ensure the year is displayed without decimals
                best_games_per_year_meta['year'] = best_games_per_year_meta['year'].astype(str)

                # Display the DataFrame in Streamlit
                st.write("Top game for each year per Metacritic")
                st.dataframe(best_games_per_year_meta[['title','platform','metascore','userscore','year']], width=750)

        with col6:

                # Group by year and find the index of the highest userscore for each year
                idx = cleaned_videogames_df.groupby('year')['userscore'].idxmax()
                # Select the rows with the highest userscore for each year
                best_games_per_year_user = cleaned_videogames_df.loc[idx]
                # Reset the index for the final output
                best_games_per_year_user.reset_index(drop=True, inplace=True)

                best_games_per_year_user['year'] = best_games_per_year_user['year'].astype(str)
                st.write("Top game for each year per Players")
                st.dataframe(best_games_per_year_user[['title', 'platform', 'metascore', 'userscore', 'year']], width=750)

        st.write("\n")
        # Correlation Heatmap
        st.write("### Correlation Heatmap without the categorical variables")
        vis.plot_correlation_heatmap(cleaned_videogames_df)
        st.markdown("""
        **Observation**: 

        1. **Metascore and Userscore**: A correlation of **0.61** indicates a relatively strong positive relationship between the scores given by critics and users. This suggests that, in general, when critics give high scores, users tend to do the same, though there is still some variability.

        2. **Year and Userscore**: There is a weak negative correlation of **-0.16** between the year and the Userscore, indicating that games released in more recent years tend to have slightly lower user ratings.

        3. **Year and Metascore**: The correlation between the year and the Metascore is very close to zero (**0.022**), indicating no significant relationship between the year of release and the critics' ratings.
        """)

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
        encoded_df = pd.concat([filtered_df[['metascore', 'userscore','year']], genre_encoded, platform_encoded], axis=1)

        # Compute the correlation matrix for all variables
        correlation_matrix = encoded_df.corr()

        # Plotting the complete correlation matrix
        plt.figure(figsize=(
        len(correlation_matrix.columns), len(correlation_matrix.columns) * 0.5))  # Size the figure to fit all variables
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=2)
        plt.title('Complete Correlation Matrix with Top 5 Genres and Platforms')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        st.pyplot(plt)

        st.markdown("""
        **Observations**: This correlation heatmap, which includes categorical variables only for top 5 genres and platforms, shows several interesting patterns:

        1. **Year correlation**: The year of release has a slight positive correlation with some platforms like **Switch** (**0.38**) and a weak negative correlation with **PC** (**-0.24**). This suggests that newer games on platforms like Switch may receive higher scores, while older games on PC may have had better performance.

        2. **Genre and Platform correlations**: There are negative correlations between certain platforms and genres, such as **PC** and **action** games (**-0.24**) and **PC** with **iOS** (**-0.45**). This suggests that certain genres may be less popular or perform less well on specific platforms.

        Overall, the heatmap reveals a mix of weak correlations, indicating that while some relationships exist between platforms, genres, and scores, they are generally not strong enough to suggest a dominant pattern.
        """)

#####################
### MACHINE LEARNING
#####################

with st.expander("**MACHINE LEARNING**"):

        st.markdown("""
                In this part, i made 2 distinct classification model:
                - **Game Success Classification:** The model try to classify the videogames in a **succesfull** game **for the players** or **not successful** knowing only the score obtained by metacritic.
                - **Quality Classification:** The model try to classify the videogames into one of these 3 categories -> [Good Game, Average Game, Bad Game] based on the selected features. The quality of the videogame in this case depends on the rating it obtained from Metacritic.
                """)

        model_selection = st.selectbox('Select which model you want to display:', ["","Game Success Classification","Quality Classification"])

        if model_selection == "Quality Classification":

                method_selected = st.selectbox("Select which classification method to use:",
                                               ["Logistic Regression", "Support Vector Machine",
                                                "Gradient Boosting", "K-Nearest Neighbors", "Random Forest"])

                features_selected = st.multiselect("Select which features to use:",
                                                   ["userscore", "platform", "genre", "year"])

                # check if there are fatures selected
                if len(features_selected) == 0:
                        st.write("Select at least one feature.")
                else:
                        X_final, y = vis.prepare_data(cleaned_videogames_df, "type1", features=features_selected)

                        options = [round(x * 0.05, 2) for x in range(1, 20)]
                        test_size = st.select_slider('Slide to select the test size', options=options, value=0.2)

                        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_size,
                                                                            random_state=22)

                        if method_selected == "Logistic Regression":
                                vis.loading_data("Logistic Regression")
                                model = LogisticRegression(random_state=22)
                        elif method_selected == "Support Vector Machine":
                                vis.loading_data("Support Vector Machine")
                                model = SVC(random_state=22)
                        elif method_selected == "Gradient Boosting":
                                vis.loading_data("Gradient Boosting")
                                model = GradientBoostingClassifier(random_state=2)
                        elif method_selected == "K-Nearest Neighbors":
                                vis.loading_data("K-Nearest Neighbors")
                                model = KNeighborsClassifier()
                        elif method_selected == "Random Forest":
                                vis.loading_data("Random Forest")
                                model = RandomForestClassifier(random_state=22)

                        # Evaluation
                        accuracy, precision, recall, f1 = vis.train_and_evaluate_model(X_train, X_test, y_train, y_test,
                                                                                       model, "type1")

                        # Display results
                        st.write(f"Accuracy: {accuracy * 100} %")
                        st.write(f"Precision: {precision * 100} %")
                        st.write(f"Recall: {recall * 100} %")
                        st.write(f"F1 Score: {f1 * 100} %")

                        vis.plot_Classification_results(accuracy, precision, recall, f1, method_selected)

        elif model_selection == "Game Success Classification":

                X_final, y = vis.prepare_data(cleaned_videogames_df, "type2")


                method_selected = st.selectbox("Select which classification method to use:",
                                               ["Logistic Regression", "Support Vector Machine",
                                                "Gradient Boosting", "K-Nearest Neighbors", "Random Forest"])

                options = [round(x * 0.05, 2) for x in range(1, 20)]
                test_size = st.select_slider('Slide to select the test size', options=options, value=0.2)

                X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size= test_size, random_state=22)

                if method_selected == "Logistic Regression":
                        vis.loading_data("Logistic Regression","type2")
                        model = LogisticRegression(random_state=22)
                elif method_selected == "Support Vector Machine":
                        vis.loading_data("Support Vector Machine","type2")
                        model = SVC(random_state=22)
                elif method_selected == "Gradient Boosting":
                        vis.loading_data("Gradient Boosting","type2")
                        model = GradientBoostingClassifier(random_state=22)
                elif method_selected == "K-Nearest Neighbors":
                        vis.loading_data("K-Nearest Neighbors","type2")
                        model = KNeighborsClassifier()
                elif method_selected == "Random Forest":
                        vis.loading_data("Random Forest","type2")
                        model = RandomForestClassifier(random_state=22)

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