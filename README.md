# Project Title: Bike Sharing Analysis Dashboard 🚴🚴

## Overview

This project is a **Bike Sharing Analysis Dashboard** that provides insights into the usage patterns of a bike-sharing system. The dashboard covers various analytical aspects such as the influence of weather on bike usage, trends across different seasons, and usage during workdays and holidays.

The dashboard is built using **Streamlit** for web-based visualization and **Jupyter Notebook** for data exploration and analysis.

## Features

- **Weather Influence Analysis**: Examine the impact of temperature, humidity, and wind speed on bike usage.
- **Seasonal Usage Trends**: Visualize the distribution of bike usage across different seasons.
- **Daily Trends**: Identify patterns in bike usage during workdays versus holidays.
- **Hourly Usage Patterns**: Heatmap visualizations to showcase peak hours for bike rentals throughout the week.
- **Extreme Weather Effects**: Analysis of how extreme weather conditions (such as heavy rain) influence bike rentals.

## Installation

To get started with this project, follow the steps below to install the necessary packages and run the dashboard.

### Prerequisites

- Python 3.8 or higher
- Streamlit Cloud (for deployment)
- Jupyter Notebook (optional for local development)

### Installation Steps

1. **Clone the repository**:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a virtual environment (optional but recommended)**:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit dashboard**:

    ```bash
    streamlit run dashboard.py
    ```

5. **Jupyter Notebook**: For exploratory data analysis, you can open the `notebook.ipynb` in Jupyter.

    ```bash
    jupyter notebook
    ```

## Requirements

These are the main dependencies for the project as listed in the `requirements.txt` file【67†source】:

```txt
pandas==2.2.3
numpy==1.26.0
matplotlib==3.9.2
seaborn==0.13.2
streamlit==1.38.0
scikit-learn
setuptools
pip==24.2
```

## How to Use

1. After running the dashboard using Streamlit, you can access the web interface in your browser.
2. The dashboard will provide the following sections:
   - **Weather Analysis**: A heatmap showing the correlation between weather parameters and bike usage.
   - **Hourly Usage**: A heatmap showing the distribution of bike usage throughout the week and during specific hours.
   - **Seasonal Analysis**: A bar chart displaying the average number of bike rentals for each season.
   - **Workday vs Holiday**: A statistical comparison of bike usage between workdays and holidays.
   - **Extreme Weather**: A comparison between bike rentals on regular days versus extreme weather conditions.

## Visualizations

Here are some visualizations you will encounter in the dashboard:

- **Heatmap** of hourly usage trends across days of the week.
- **Bar plots** comparing seasonal and workday/holiday bike usage.
- **Line charts** showing temperature and its impact on the number of bike users.
  
## Dataset

The dataset used in this analysis includes hourly data for bike rentals and weather conditions. It features columns like:
- `cnt`: Count of bike rentals.
- `temp`: Temperature normalized.
- `hum`: Humidity levels.
- `windspeed`: Wind speed.
- `weathersit`: Weather conditions (e.g., Clear, Cloudy, Rain).
- `season`: Seasons (Winter, Spring, Summer, Fall).
- `weekday`: Day of the week.
