import time
import requests
import streamlit as st
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import google.generativeai as genai
from transformers import pipeline
from deepface import DeepFace

# 🔹 Configure Google Gemini API
genai.configure(api_key="YOUR_API_KEY") #Replace tour gemini api key here

# 🔹 Convert shorthand numbers
def convert_shorthand_number(n):
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    if isinstance(n, str) and n[-1] in multipliers:
        return int(float(n[:-1]) * multipliers[n[-1]])
    return int(n)

# 🔹 Initialize sentiment and summarization pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

# 🔹 Analyze tweet sentiment and generate summary
def analyze_tweet(tweet_text):
    sentiment_result = sentiment_analyzer(tweet_text)[0]
    sentiment_label = sentiment_result["label"]

    if len(tweet_text)<10:
        summary = "N/A"
    else:
        try:
            summary = summarizer(tweet_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        except:
            summary = "Summary not available"

    return sentiment_label, summary

# 🔹 Initialize WebDriver
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# 🔹 Download and process tweet images
def download_image(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    return None

# 🔹 Perform DeepFace analysis on images
def analyze_face(image_url):
    image = download_image(image_url)
    if image is None:
        return "No face detected or image not available."

    try:
        analysis = DeepFace.analyze(image, actions=["age", "gender", "emotion", "race"], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]

        age = analysis["age"]
        gender = analysis["dominant_gender"]
        emotion = analysis["dominant_emotion"]
        race = analysis["dominant_race"]

        return f"👤 Age: {age}, Gender: {gender}, Emotion: {emotion}, Race: {race}"
    except Exception as e:
        return f"Error analyzing face: {e}"

# 🔹 Scrape tweets and extract images
def scrape_twitter_profile(profile_url):
    driver = init_driver()
    driver.get(profile_url)
    time.sleep(5)

    for _ in range(3):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        time.sleep(2)

    tweets_data = []
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//article[@data-testid='tweet']"))
        )
        tweet_elements = driver.find_elements(By.XPATH, "//article[@data-testid='tweet']")

        for tweet in tweet_elements[:10]:
            engagement_divs = tweet.find_elements(By.XPATH, ".//div[@role='group']/div")

            try:
                tweet_text = tweet.find_element(By.XPATH, ".//div[@data-testid='tweetText']").text.strip()
            except:
                tweet_text = "N/A"

            try:
                image_element = tweet.find_element(By.XPATH, ".//img[@alt='Image']")
                image_url = image_element.get_attribute("src")
            except:
                image_url = None

            try:
                likes = engagement_divs[2].find_element(By.XPATH, ".//span").text.strip() or "0"
                likes = convert_shorthand_number(likes)
            except:
                likes = 0

            try:
                comments = engagement_divs[0].find_element(By.XPATH, ".//span").text.strip() or "0"
                comments = convert_shorthand_number(comments)
            except:
                comments = 0

            sentiment, summary = analyze_tweet(tweet_text)
            face_analysis = analyze_face(image_url) if image_url else "No image available"

            tweets_data.append({
                "Tweet": tweet_text,
                "Likes": likes,
                "Comments": comments,
                "Image_URL": image_url,
                "Summary": summary,
                "Sentiment": sentiment,
                "Face_Analysis": face_analysis
            })
    except Exception as e:
        print(f"Error: {e}")

    driver.quit()
    return tweets_data

# 🔹 Generate AI-based insights using Google Gemini
def generate_ai_reasoning(tweets):
    if not tweets:
        return []

    most_liked_tweet = max(tweets, key=lambda x: x["Likes"])
    most_liked_text = most_liked_tweet["Tweet"]
    most_liked_likes = most_liked_tweet["Likes"]

    insights = []
    for tweet in tweets:
        if tweet["Likes"] == most_liked_likes:
            insights.append("This is the most liked tweet.")
            continue

        prompt = f"""
        Analyze why this tweet received fewer engagements than the most liked tweet.

        Most Liked Tweet:
        - Text: {most_liked_text}
        - Likes: {most_liked_likes}

        Tweet to Compare:
        - Text: {tweet['Tweet']}
        - Likes: {tweet['Likes']}
        - Comments: {tweet['Comments']}
        - Sentiment: {tweet['Sentiment']}
        - Face Analysis: {tweet['Face_Analysis']}

        Provide a detailed analysis of possible reasons.
        """

        model = genai.GenerativeModel("models/gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        ai_reasoning = response.text if response.text else "No insights available."

        insights.append(ai_reasoning)

    return insights

# 🔹 Streamlit UI
st.title("🐦 Twitter (X) Profile Analyzer with AI & DeepFace")
profile_link = st.text_input("🔗 Enter Twitter Profile Link (e.g., https://twitter.com/elonmusk)")

if st.button("Submit"):
    if profile_link:
        tweets = scrape_twitter_profile(profile_link)
        insights = generate_ai_reasoning(tweets)

        st.markdown("### 🐦 Recent Tweets:")

        if tweets:
            for i, tweet in enumerate(tweets):
                with st.expander(f"📢 {tweet['Tweet']}"):
                    st.write(f"**Summary:** {tweet['Summary']}")
                    st.write(f"**AI Analysis:** {insights[i]}")
                    st.write(f"**Face Analysis:** {tweet['Face_Analysis']}")
                    if tweet["Image_URL"]:
                        st.image(tweet["Image_URL"], caption="Tweet Image")
        else:
            st.write("No tweets found or unable to fetch data.")
