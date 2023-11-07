import streamlit as st
import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
import gensim
import pyLDAvis.gensim_models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from streamlit.components.v1 import html
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import base64
from io import BytesIO
import numpy as np
from PIL import Image

# Download necessary nltk data
nltk.download(['stopwords', 'vader_lexicon'])

# Initialize the sentiment analysis pipeline
sentiment_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Streamlit App UI
st.title('Amazon Reviews Insights ✏️📈')

# Sidebar UI
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type="xlsx")
analysis_options = [
    'Word Cloud 🌥️',
    'Positive/Negative Word Cloud 😊😠',
    'Emotion Analysis 🎭',
    'NPS Analysis 📊',
    'Intent Analysis 🎯',
    'Top Bigrams Analysis 🔠',
    'Sentiment Over Time ⏳'
]
analysis_choice = st.sidebar.selectbox('Choose an analysis:', options=analysis_options)

# Function to load and preprocess data
def load_and_preprocess_data(file):
    df = pd.read_excel(file, engine='openpyxl')
    df['full_review'] = df['Title'].astype(str) + " " + df['Body'].astype(str)
    return df

# Function to display LDA visualization
def display_lda_visualization(html_filename):
    with open(html_filename, 'r', encoding='utf-8') as file:
        html_content = file.read()
        html(html_content, width=900, height=800, scrolling=True)

def get_image_download_link(img, filename="wordcloud.png", text="Download WordCloud image"):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def plot_top_words(texts, ngram_range=(1, 1), top_n=10, title="Top N-grams"):

    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0)
    data = []
    for col, term in sorted(enumerate(vectorizer.get_feature_names_out()), key=lambda col_term: col_term[1]):
        data.append((term, sums[0, col]))
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]

    terms, counts = zip(*sorted_data)
    indices = np.arange(len(terms))

    # Use seaborn to improve aesthetics
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=list(counts), y=list(terms), palette='coolwarm', ax=ax)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Occurrences', fontsize=14)
    ax.set_ylabel('Bigrams', fontsize=14)
    ax.set_yticklabels(terms, fontsize=12)
    
    # Show the count on the bars
    for i, count in enumerate(counts):
        ax.text(count, i, f' {count}', ha='left', va='center', fontsize=12)

    # Remove spines
    sns.despine(left=True, bottom=True)

    # Add some padding between the axis and the labels
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

    # Tight layout often produces nicer spacing between subplots
    plt.tight_layout()

    # Use Streamlit to display the figure
    st.pyplot(fig)

    
def plot_wordcloud(texts, ngram_range=(1, 1), title="Word Cloud"):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(texts)
    word_counts = X.sum(axis=0).A1
    word_frequencies = {word: word_counts[idx] for word, idx in vectorizer.vocabulary_.items()}
    
    wordcloud = WordCloud(width=1600, height=800, background_color='white')
    wordcloud.generate_from_frequencies(word_frequencies)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

def analyze_emotions(data):
    data['full_review'] = data['Title'].astype(str) + ". " + data['Body'].astype(str)
    data['emotion'] = data['full_review'].apply(lambda x: sentiment_classifier(x[:512])[0]['label'])
    emotion_counts = data['emotion'].value_counts()
    return data, emotion_counts


# Main Page UI and Logic
if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    stop_words = set(stopwords.words('english'))
    sid = SentimentIntensityAnalyzer()

    # Word Cloud Analysis
    if analysis_choice == 'Word Cloud 🌥️':
        st.subheader('Monogram Word Cloud')
        texts = df['full_review'].dropna().tolist()
        plot_wordcloud(texts, ngram_range=(1, 1), title="Monogram Word Cloud")

        st.subheader('Bigram Word Cloud')
        plot_wordcloud(texts, ngram_range=(2, 2), title="Bigram Word Cloud")




    # Positive/Negative Word Cloud Analysis
    elif analysis_choice == 'Positive/Negative Word Cloud 😊😠':
        st.subheader('Positive Word Cloud 🌤️')
        df['sentiment_score'] = df['full_review'].apply(lambda review: sid.polarity_scores(review)['compound'])
    
    # Separate positive and negative reviews
        positive_reviews = df[df['sentiment_score'] > 0]['full_review']
        negative_reviews = df[df['sentiment_score'] < 0]['full_review']
    
    # Generate word clouds using bigrams
        pos_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        pos_bigram_matrix = pos_vectorizer.fit_transform(positive_reviews)
        pos_frequencies = sum(pos_bigram_matrix).toarray()[0]
        pos_bigram_frequencies = {bigram: freq for bigram, freq in zip(pos_vectorizer.get_feature_names_out(), pos_frequencies)}
    
        pos_wordcloud = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(pos_bigram_frequencies)
    
        fig1, ax1 = plt.subplots()
        ax1.imshow(pos_wordcloud, interpolation='bilinear')
        ax1.set_title('Positive Reviews Bigrams')
        ax1.axis("off")
        st.pyplot(fig1)  # Show the positive bigram word cloud
    
        st.subheader('Negative Word Cloud 🌧️')
        neg_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        neg_bigram_matrix = neg_vectorizer.fit_transform(negative_reviews)
        neg_frequencies = sum(neg_bigram_matrix).toarray()[0]
        neg_bigram_frequencies = {bigram: freq for bigram, freq in zip(neg_vectorizer.get_feature_names_out(), neg_frequencies)}
    
        neg_wordcloud = WordCloud(width=1600, height=800, background_color='black').generate_from_frequencies(neg_bigram_frequencies)
    
        fig2, ax2 = plt.subplots()
        ax2.imshow(neg_wordcloud, interpolation='bilinear')
        ax2.set_title('Negative Reviews Bigrams')
        ax2.axis("off")
        st.pyplot(fig2)  # Show the negative bigram word cloud


    # Emotion Analysis
    elif analysis_choice == 'Emotion Analysis 🎭':
        st.subheader('Emotion Analysis 🎭')
        # Bulk emotion analysis
        if st.button('Analyze Emotions in Reviews'):
            with st.spinner('Analyzing...'):
                analyzed_data, emotion_counts = analyze_emotions(df)
                st.write(analyzed_data[['Title', 'Body', 'emotion']])
                st.success('Analysis complete!')
                st.subheader('Emotion Counts')
                st.write(emotion_counts)  # This line displays the emotion counts




    # NPS Analysis
    elif analysis_choice == 'NPS Analysis 📊':
        st.subheader('Net Promoter Score (NPS) Analysis 📊')
        df['sentiment_score'] = df['full_review'].apply(lambda review: sid.polarity_scores(review)['compound'])
    
    # Calculate NPS labels
        df['NPS_label'] = df['sentiment_score'].apply(lambda score: 'Promoter' if score > 0.05 else ('Detractor' if score < -0.05 else 'Passive'))
        nps = (len(df[df['NPS_label'] == 'Promoter']) - len(df[df['NPS_label'] == 'Detractor'])) / len(df) * 100
        st.write(f"Net Promoter Score: **{nps:.2f}**")
    
        st.write("""
- **NPS < 0**: This indicates that there are more detractors than promoters, which means that improvements might be necessary to enhance customer satisfaction.
- **0 ≤ NPS ≤ 30**: A positive NPS indicates that there are more promoters than detractors, but there is room for improvement.
- **30 < NPS ≤ 70**: This is considered a good NPS range, indicating a significant number of promoters and a customer-centric business.
- **NPS > 70**: An excellent NPS, indicating a high level of customer loyalty and satisfaction.
""")
        st.subheader("Bar Chart for Detractors and Promoters")
    
    # Show bar chart and counts of promoters, passives, and detractors
        nps_counts = df['NPS_label'].value_counts()
        st.bar_chart(nps_counts)
    
    # Print the counts below the chart
        st.write(f"Number of Promoters: {nps_counts.get('Promoter', 0)}")
        st.write(f"Number of Passives: {nps_counts.get('Passive', 0)}")
        st.write(f"Number of Detractors: {nps_counts.get('Detractor', 0)}")


    # Intent Analysis
    elif analysis_choice == 'Intent Analysis 🎯':
        st.subheader('Intent Analysis 🎯')

        df['sentiment_score'] = df['full_review'].apply(lambda review: sid.polarity_scores(review)['compound'])

        def assign_intent_label(score):
            if score >= 0.05:
                return 'Positive'
            elif score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'

        df['Intent'] = df['sentiment_score'].apply(assign_intent_label)
        

        positive_intent_count = len(df[df['Intent'] == 'Positive'])
        negative_intent_count = len(df[df['Intent'] == 'Negative'])
        neutral_intent_count = len(df[df['Intent'] == 'Neutral'])
        
        
        st.subheader("Bar Chart for the Intent Analysis-")
        intent_counts = df['Intent'].value_counts()
        st.bar_chart(intent_counts)
        
        st.write(f"Number of Positive Intents: {positive_intent_count}")
        st.write(f"Number of Neutral Intents: {neutral_intent_count}")
        st.write(f"Number of Negative Intents: {negative_intent_count}")
        
        st.write(df[['full_review', 'Intent']])


    # Top Bigrams Analysis
    elif analysis_choice == 'Top Bigrams Analysis 🔠':
        st.subheader('Top Bigrams Analysis 🔠')
        
        plot_top_words(df['full_review'].dropna(), ngram_range=(2, 2), top_n=10, title="Top 2-word phrases in reviews")

    # Sentiment Over Time Analysis
    elif analysis_choice == 'Sentiment Over Time ⏳':
        st.subheader('Sentiment Over Time ⏳')
        # Assuming there's a date column 'Review_Date' in the dataframe
        if 'Review_Date' in df.columns:
            df['Review_Date'] = pd.to_datetime(df['Review_Date'])
            df.set_index('Review_Date', inplace=True)
            df['sentiment_score'] = df['full_review'].apply(lambda review: sid.polarity_scores(review)['compound'])
            daily_sentiment = df.resample('D')['sentiment_score'].mean().fillna(0)
            
            plt.figure(figsize=(10,5))
            sns.lineplot(data=daily_sentiment)
            plt.title('Daily Sentiment Trend')
            plt.xlabel('Date')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot()
        else:
            st.error("The uploaded Excel file does not contain a 'Review_Date' column.")



else:
    st.warning('Please upload a file to enable the analysis options.')
