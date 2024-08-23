# 🎮 Videogames: Metacritic vs Userscore Ratings

This dataset contains a collection of video games released between **1998 and 2021**, featuring ratings from both **critics** (via Metacritic) and **players** (Userscore). The dataset is particularly useful for comparing how critics and players rate games across different **genres**, **platforms**, and **time periods**.

## 📊 Dataset Features

- **Titles**: The names of the video games.
- **Platforms**: The platforms on which the games are available (e.g., PC, PlayStation 4, Switch). Some games are released on multiple platforms.
- **Metascore**: The aggregated critic score from [Metacritic](https://www.metacritic.com), on a scale from 0 to 100.
- **Userscore**: The player score, provided by users on Metacritic, on a scale from 0 to 10. For some new or less popular games, the score may still be marked as 'tbd' (to be determined).
- **Genre**: The genre of the game (e.g., action, adventure, RPG). Some games may fit into multiple genres.
- **Date**: The official release date of the game.

---

## 🧠 Analysis Focus

The analysis performed on this dataset focuses on several key areas:

1. **Average Ratings by Genre and Platform**:
   - How do **critic** and **player ratings** vary across different game **genres** (e.g., action, RPG, adventure)?
   - How do ratings differ based on the **platform** (e.g., PC, PlayStation, Xbox) the game was released on?

2. **Distribution Comparison**:
   - A comparison of the **distribution of average critic and player scores**, highlighting potential trends or differences between how critics and players rate games.

3. **Yearly Trend Analysis**:
   - Exploring the **average ratings** of critics and players over time, to observe how these ratings have evolved from **1998 to 2021**.

4. **Classification System: Game Quality**:
   - A **classification model** is built to predict whether a game is classified as **Good**, **Average**, or **Bad** based on selected features (e.g., platform, genre, userscore, year of release). This model allows for an exploration of what factors contribute most to a game's success or failure, according to the ratings.

5. **Classification System: Game Success Among Players**:
   - Another classification model is built to predict whether a game was considered **successful** among players based **only on the Metascore**. This analysis provides insight into the correlation between critic scores and player reception, exploring whether a higher Metascore is a reliable predictor of a game's success among its audience.

---

This analysis provides valuable insights into the relationship between **critic and player ratings**, allowing a deeper understanding of how games are perceived across different genres, platforms, and over time. The two classification systems—one for **game quality** and the other for **player success**—add a predictive element, showing how certain features and ratings may influence a game's perception and success.
