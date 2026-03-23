NewsPulse — Global News Trend Analyzer
Infosys Springboard Virtual Internship Project

About the Project
NewsPulse is a real-time news analytics web application built using Python and Streamlit. The idea behind this project was to collect live news articles from across the world and apply NLP and machine learning techniques to find patterns, trending topics, and the overall sentiment of the news.

The application fetches news from the NewsAPI, cleans and processes the text, runs it through a machine learning pipeline, and displays everything in an interactive dashboard with charts and visualizations.

Features
Real-time news fetching using NewsAPI
Text cleaning and preprocessing using NLTK
Sentiment analysis (Positive, Negative, Neutral) using VADER
Keyword extraction using TF-IDF
Topic modeling using LDA (Latent Dirichlet Allocation)
Article clustering using K-Means
Interactive charts and visualizations using Plotly
Role-based login system (Analyst, Editor, Admin)
Article search and filter
CSV export of analyzed data
Admin panel for data quality checks
Tech Stack
Component	Technology Used
Web Framework	Streamlit
News Data	NewsAPI + requests
Data Handling	Pandas, NumPy
NLP & Preprocessing	NLTK
Sentiment Analysis	NLTK VADER
Keyword Extraction	Scikit-learn TF-IDF
Topic Modeling	Scikit-learn LDA
Clustering	Scikit-learn K-Means
Visualization	Plotly
Project Structure
NewsPulseProject/
│
├── app.py              → Main application file
├── requirements.txt    → Required Python libraries
└── README.md           → Project documentation
How to Run
Step 1 — Install the required libraries:

pip install -r requirements.txt
Step 2 — Run the application:

python -m streamlit run app.py
Step 3 — Open in browser:

http://127.0.0.1:8501
Login Details
Username	Password	Role
analyst	pulse2024	Analyst
editor	editor123	Editor
admin	admin999	Admin
How It Works
1. Data Collection
News articles are fetched using the NewsAPI. The application supports two modes — browsing by category (technology, business, health, etc.) or searching by custom keywords with a date range. Each article collected includes the title, description, source, publication date, and category.

2. Text Preprocessing
Before applying any machine learning, the raw text is cleaned. This involves removing HTML tags, URLs, special characters, and extra whitespace. The text is then tokenized using NLTK, stopwords are removed, and each word is lemmatized to its base form. This step is important because cleaner text gives better results in the NLP models.

3. Sentiment Analysis
Sentiment is calculated using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner). It gives a compound score between -1 and +1 for each article. Articles with a score above 0.05 are labeled Positive, below -0.05 are Negative, and everything in between is Neutral.

4. TF-IDF Keyword Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) is used to find the most important keywords across all the articles. Words that appear frequently in a specific article but not across all articles are ranked higher. This helps identify what topics are trending.

5. LDA Topic Modeling
LDA (Latent Dirichlet Allocation) is an unsupervised machine learning algorithm that discovers hidden topics in a collection of documents. Each topic is represented by a group of related keywords. The number of topics can be adjusted from the sidebar.

6. K-Means Clustering
K-Means is used to group similar articles together into clusters. The TF-IDF vectors of the articles are used as input features, and the algorithm groups them based on similarity. Each cluster is named using its most representative keywords.

7. Visualization
All results are displayed using Plotly interactive charts including bar charts, line graphs, pie charts, histograms, and treemaps. The dashboard has a dark teal theme for a modern look.

Dashboard Tabs
Overview — Summary stats, articles by category, top sources, keyword importance
Trends — Daily article volume, sentiment over time, category breakdown
Topics — LDA topic cards with keyword tags, K-Means cluster charts
Sentiment — Sentiment distribution, score histogram, most positive and negative articles
Articles — Searchable and filterable article browser with CSV download
Admin — Data quality checks and raw data preview (admin login only)
Notes
A default NewsAPI key is included in the code for testing purposes
For production use, replace it with your own key from https://newsapi.org
The free NewsAPI plan allows up to 100 requests per day
Python 3.11 is recommended for running this project
