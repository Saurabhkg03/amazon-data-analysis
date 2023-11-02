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
import base64
from io import BytesIO

# Download necessary nltk data
nltk.download(['stopwords', 'vader_lexicon'])

# Streamlit App UI
st.title('Amazon Reviews Insights')

# Sidebar UI
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type="xlsx")
analysis_options = [
    'Word Cloud',
    'Positive/Negative Word Cloud',
    'LDA Analysis',
    'NPS Analysis',
    'Intent Analysis',
    'Sentiment Over Time'
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

# Main Page UI and Logic
if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    stop_words = set(stopwords.words('english'))
    sid = SentimentIntensityAnalyzer()

    # Word Cloud Analysis
    if analysis_choice == 'Word Cloud':
        st.subheader('Word Cloud')
        texts = " ".join(review for review in df['full_review'])
        wordcloud = WordCloud(width=1600, height=800, max_words=150, stopwords=stop_words, background_color='white').generate(texts)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        st.markdown(get_image_download_link(wordcloud.to_image()), unsafe_allow_html=True)

    # Positive/Negative Word Cloud Analysis
    elif analysis_choice == 'Positive/Negative Word Cloud':
        st.subheader('Positive/Negative Word Cloud')
        df['sentiment_score'] = df['full_review'].apply(lambda review: sid.polarity_scores(review)['compound'])
        
        # Separate positive and negative reviews
        positive_reviews = df[df['sentiment_score'] > 0]['full_review']
        negative_reviews = df[df['sentiment_score'] < 0]['full_review']
        
        # Generate word clouds
        pos_wordcloud = WordCloud(width=800, height=400, max_words=150, stopwords=stop_words, background_color='white').generate(" ".join(positive_reviews))
        neg_wordcloud = WordCloud(width=800, height=400, max_words=150, stopwords=stop_words, background_color='black').generate(" ".join(negative_reviews))

        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        axs[0].imshow(pos_wordcloud, interpolation='bilinear')
        axs[0].set_title('Positive Reviews')
        axs[0].axis("off")
        axs[1].imshow(neg_wordcloud, interpolation='bilinear')
        axs[1].set_title('Negative Reviews')
        axs[1].axis("off")
        st.pyplot(fig)

    # LDA Analysis
    elif analysis_choice == 'LDA Analysis':
        st.subheader('LDA Analysis')
        # Preprocess reviews for LDA
        def preprocess_review(review):
            tokens = word_tokenize(review)
            tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
            return tokens
        
        df['tokens'] = df['full_review'].apply(preprocess_review)
        texts = df['tokens']
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)
        for idx, topic in lda_model.print_topics(-1):
            st.write('Topic: {} \nWords: {}'.format(idx, topic))
        
        lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(lda_display, 'lda_visualization.html')
        if st.button('Show LDA Visualization'):
            display_lda_visualization('lda_visualization.html')

    # NPS Analysis
    elif analysis_choice == 'NPS Analysis':
        st.subheader('Net Promoter Score (NPS) Analysis')
        df['sentiment_score'] = df['full_review'].apply(lambda review: sid.polarity_scores(review)['compound'])
        
        # Calculate NPS
        df['NPS_label'] = df['sentiment_score'].apply(lambda score: 'Promoter' if score > 0.05 else ('Detractor' if score < -0.05 else 'Passive'))
        nps = (len(df[df['NPS_label'] == 'Promoter']) - len(df[df['NPS_label'] == 'Detractor'])) / len(df) * 100
        st.write(f"Net Promoter Score: {nps:.2f}")
        st.write("""
- **NPS < 0**: This indicates that there are more detractors than promoters, which means that improvements might be necessary to enhance customer satisfaction.
- **0 ≤ NPS ≤ 30**: A positive NPS indicates that there are more promoters than detractors, but there is room for improvement.
- **30 < NPS ≤ 70**: This is considered a good NPS range, indicating a significant number of promoters and a customer-centric business.
- **NPS > 70**: An excellent NPS, indicating a high level of customer loyalty and satisfaction.
""")
        st.subheader("Bar Chart for Detractors and Promoters")
        # Show counts of promoters, passives, and detractors
        st.bar_chart(df['NPS_label'].value_counts())

    # Intent Analysis
    elif analysis_choice == 'Intent Analysis':
        st.subheader('Intent Analysis')
        df['sentiment_score'] = df['full_review'].apply(lambda review: sid.polarity_scores(review)['compound'])
        
        df['Intent'] = df['sentiment_score'].apply(lambda score: 'Positive' if score > 0.05 else ('Negative' if score < -0.05 else 'Neutral'))
        st.write(df[['full_review', 'Intent']])
        
        intent_counts = df['Intent'].value_counts()
        st.bar_chart(intent_counts)

    elif analysis_option == 'Top Bigrams Analysis':
        st.subheader('Top Bigrams Analysis')
        
        plot_top_words(df['full_review'].dropna(), ngram_range=(2, 2), top_n=10, title="Top 2-word phrases in reviews")


    # Sentiment Over Time Analysis
    elif analysis_choice == 'Sentiment Over Time':
        st.subheader('Sentiment Over Time')
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
