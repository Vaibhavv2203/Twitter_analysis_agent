# Twitter (X) Profile Analyzer with AI and DeepFace

An AI-powered Twitter (X) profile analysis system built using Streamlit, Selenium, DeepFace, Hugging Face Transformers, and Google Gemini AI.

The application scrapes tweets from Twitter profiles, analyzes tweet engagement, performs sentiment analysis, summarizes tweet content, detects facial attributes from tweet images, and generates AI-driven insights explaining tweet performance.

---

## Features

- Twitter (X) profile scraping
- Tweet sentiment analysis
- AI-generated tweet summaries
- DeepFace image analysis
- Tweet engagement analysis
- AI-powered reasoning for tweet performance
- Image extraction from tweets
- Automated tweet comparison
- Streamlit interactive dashboard

---

## Tech Stack

- Python
- Streamlit
- Selenium
- Google Gemini AI
- Hugging Face Transformers
- DeepFace
- OpenCV
- BeautifulSoup
- Pandas
- NumPy

---

## Project Workflow

1. User enters a Twitter (X) profile link.
2. Selenium scrapes recent tweets from the profile.
3. The system extracts:
   - Tweet text
   - Likes
   - Comments
   - Tweet images

4. Hugging Face models perform:
   - Sentiment analysis
   - Tweet summarization

5. DeepFace analyzes images for:
   - Age
   - Gender
   - Emotion
   - Race

6. Gemini AI compares tweet performance and generates reasoning for engagement differences.

7. Results are displayed in an interactive Streamlit interface.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/twitter-profile-analyzer.git
cd twitter-profile-analyzer
