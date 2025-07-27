# Nepali Music Analytics & Prediction with Audio Features

This project is a data-driven analysis and machine learning toolkit for exploring and understanding patterns, and styles within Nepali music using Spotify-derived audio features.

It combines exploratory data analysis, visualizations, unsupervised learning, and supervised machine learning models to estimate popularity, all using audio feature vectors.

---

## Features Analyzed

The dataset contains audio features of 2000+ Nepali songs scraped using Spotify

- `BPM` (Tempo)
- `Energy`
- `Danceability`
- `Loudness`
- `Valence` (Mood)
- `Length`
- `Acousticness`
- `Popularity`

## Analysis & Visualizations of Nepali Music

Implemented with `Seaborn`, `Matplotlib`, and `Plotly`:

- **Sunburst Chart** – Visual hierarchy of `Energy` and `Valence`
- **Radar Chart** – Comparison of audio profiles across simulated genres
- **3D Scatter Plot** – Visualization of `BPM`, `Danceability`, and `Energy`
- **2D Scatter Plot (Valence vs Energy)** – Colored by Simulated Genre
- **Joint Plot (Valence vs Energy)** – KDE and regression overlays
- **Bubble Chart** – Size represents `Popularity`, axes `Valence` and `Energy`
- **Boxplots** – For `Valence`, `Energy`, `Dance`, `Acoustic` across genres
- **Violin Plot** – Acousticness variation across simulated genres
- **Distribution Plots** – For all individual features (BPM, Energy, Dance, etc.)
- **PairGrid** – Multivariate scatterplot matrix with genre hue
- **Correlation Heatmap** – To show feature interdependencies
- **Cumulative Acousticness Trend** – Line plot for understanding acoustic nature
- **Song Length Distribution** – Histogram of song durations
- **Energy Histogram** – Intensity of songs
- **Average Valence Over Time** – Using `Release Date` for mood trends
- **Danceability vs Valence Regression** – Trend between mood and dance features
- **Feature Importance Plot** – Based on model predicting popularity

---

## Machine Learning Models

Trained using Scikit-learn:

| Task                  | Model            | Description               |
| --------------------- | ---------------- | ------------------------- |
| Popularity Prediction | Ridge Regression | Estimate popularity score |

Models are saved in `models/` as `.pkl` files.

## Other models tested but not used in final deployment:

**Linear Regression**
**Random Forest Regressor**
**SVR**

---

## Web App (Streamlit)

A simple web interface is available at `nepali_music_popularity_prediction_app.py`:

- Predicts popularity of a song
- Sliders to input audio features
- Interactive and lightweight

---

### Run it:

```bash
streamlit run nepali_music_popularity_prediction_app.py


```
