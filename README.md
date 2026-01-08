
# 📚 Audible Insights: NLP Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fohh29-audibleinsights-recommender-app-XXXXXX.streamlit.app)

> **Live Demo:** [Click here to view the app](https://fohh29-audibleinsights-recommender-app-XXXXXX.streamlit.app)

---

## 🛠️ Project Deliverables & Implementation

### 1. Data Preparation
* **Merging:** Combined `Audible_Catlog.csv` and `Audible_Catlog_Advanced_Features.csv` using a left-join on 'Book Name' and 'Author'.
* **Cleaning:** Handled missing ratings, converted price data, and extracted primary genres from messy categorical strings.

### 2. NLP and Clustering
* **Feature Extraction:** Used `TF-IDF Vectorization` (Term Frequency-Inverse Document Frequency) to convert book descriptions into high-dimensional vectors.
* **Method:** Implemented `K-Means Clustering` to group books into 10 distinct thematic clusters.
* **Similarity:** Used `Cosine Similarity` to calculate the distance between books for precise recommendations.

### 3. Model Performance
* **Precision:** Evaluated recommendations based on genre consistency and thematic relevance.
* **Insights:** Discovered that clusters accurately group "Self-Help" and "Business" books even when labeled differently in the raw data.

### 4. Application Functionality
* Developed a responsive **Streamlit** interface that allows users to search for books and discover "Hidden Gems" (high-rated books with low popularity).

---

## 🚀 How to Run Locally
1. Clone the repo: `git clone https://github.com/fohh29/AudibleInsights-Recommender.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`