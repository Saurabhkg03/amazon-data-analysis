# Amazon Reviews Data Analysis App

This repository contains a Streamlit application designed to perform sentiment analysis and other insightful analytics on Amazon product reviews.

## Description

The Streamlit app allows users to upload an Excel file containing Amazon reviews and choose from various types of analysis, including word clouds, LDA topic modeling, Net Promoter Score (NPS) analysis, intent analysis, and sentiment over time. The app is interactive and provides visualizations for better insight into customer sentiments and opinions.

## Analysis Types

- **Word Cloud**: Generates a word cloud from all the reviews to visualize the most frequently occurring words.
- **Positive/Negative Word Cloud**: Creates separate word clouds for positive and negative reviews to easily identify common themes in each sentiment category.
- **Emotion Analysis**: Emotion analysis, also known as sentiment analysis or sentiment detection, is a natural language processing (NLP) technique used to determine the emotional tone or sentiment expressed in text data.
- **LDA Analysis**: Performs Latent Dirichlet Allocation (LDA) to discover the main topics that appear in the reviews.
- **NPS Analysis**: Calculates the Net Promoter Score to gauge customer loyalty and satisfaction based on their reviews.
- **Intent Analysis**: Categorizes reviews into 'Positive', 'Negative', or 'Neutral' based on the sentiment score.
- **Sentiment Over Time**: Plots the average sentiment score of reviews over time to analyze trends.


## Installation

To run this app, you need Python installed on your system. If you do not have Python installed, please follow the instructions on the [Python Downloads](https://www.python.org/downloads/) page. Once Python is installed, you can set up a virtual environment and install the required libraries.

1. Clone this repository:

    ```
    git clone https://github.com/Saurabhkg03/amazon-data-analysis.git
    cd amazon-data-analysis
    ```

2. (Optional) Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required libraries:

    ```bash
     pip install streamlit pandas nltk wordcloud matplotlib seaborn gensim pyLDAvis scikit-learn transformers numpy Pillow tensorflow torch==2.1.0 torchvision==0.16.0 openpyxl
    ```

## Usage

To run the app, navigate to the cloned directory in your terminal and run the following command:

    
    streamlit run app.py
    

Replace `app.py` with the actual name of the Python script.

Once the app is running, follow these steps:

1. Open your web browser and go to `http://localhost:8501`.
2. Use the sidebar to upload an Excel file containing Amazon reviews.
3. Select the type of analysis you want to perform from the dropdown menu.
4. View the analysis and visualizations that appear on the main page.


## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
