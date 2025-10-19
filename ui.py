import streamlit as st
import pandas as pd
from all_fun import *
import re
from nltk.tokenize import word_tokenize

STOPWORDS_PATH = r"C:\Users\akshi\Downloads\Nlp_prj_2\stopwords.txt"
POSITIVE_PATH = r"C:\Users\akshi\Downloads\Nlp_prj_2\positive_words.txt"
NEGATIVE_PATH = r"C:\Users\akshi\Downloads\Nlp_prj_2\negative_words.txt"
st.set_page_config(layout='wide')

@st.cache_data
def load_wordlists():
    stopwords_list = load_word_list(STOPWORDS_PATH)
    positive_list = load_word_list(POSITIVE_PATH)
    negative_list = load_word_list(NEGATIVE_PATH)
    return stopwords_list, positive_list, negative_list

stopwords_list, positive_list, negative_list = load_wordlists()


st.sidebar.title("📂 Controls")
url = st.sidebar.text_input("Enter a URL:")
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file with URLs", type=["csv"])

st.title("🔍 Sentiment & Readability Analysis")


def analyze_single_url(url):
    title, content = scrape_url(url)
    if not content:
        return None  # skip if no content

    cleaned_content = clean_text(content)
    df = pd.DataFrame([{
        "URL": url,
        "Title": title,
        "Cleaned Content": cleaned_content
    }])

    sentiment_res = process(df, stopwords_list, positive_list, negative_list)[0]

    avg_sent_len, pct_complex_words, fog_index = compute_readability_metrics(cleaned_content)
    total_words, total_sentences, avg_words_per_sent = calculate_metrics(cleaned_content)
    complex_words_count = complex_words(cleaned_content)

    words = word_tokenize(cleaned_content)
    cleaned_words = [re.sub(r'[^\w\s]', '', w.lower()) for w in words if w.lower() not in stopwords_list]
    cleaned_words = [w for w in cleaned_words if w]
    total_syllables, syllable_list = analyze_syllables(cleaned_content)
    avg_syllables = total_syllables / len(syllable_list) if syllable_list else 0
    pronouns = count_personal_pronouns(cleaned_content)
    avg_word_len = calculate_average_word_length(cleaned_content)

    return {
        "URL": url,
        "Title": title,
        **sentiment_res,
        "Average Sentence Length": round(avg_sent_len, 2),
        "Percentage Complex Words": round(pct_complex_words, 2),
        "Fog Index": round(fog_index, 2),
        "Total Words": total_words,
        "Total Sentences": total_sentences,
        "Average Words per Sentence": avg_words_per_sent,
        "Number of Complex Words": complex_words_count,
        "Cleaned Word Count": len(cleaned_words),
        "Total Syllables": total_syllables,
        "Average Syllables per Word": round(avg_syllables, 2),
        "Personal Pronouns": pronouns,
        "Average Word Length": round(avg_word_len, 2)
    }


if url and not uploaded_file:
    result = analyze_single_url(url)
    if result:
        col1, col2 = st.columns([2, 2])
        with col1:
            st.subheader("Page Information")
            st.write(f"**Title:** {result['Title']}")
            st.write(f"**URL:** {result['URL']}")
            st.write(result["Cleaned Content"] if "Cleaned Content" in result else "Content hidden in metrics.")
        with col2:
            st.subheader("Sentiment Analysis")
            st.write(f"Positive Count: {result['positive_count']}")
            st.write(f"Negative Count: {result['negative_count']}")
            st.write(f"Polarity: {result['polarity']}")
            st.write(f"Subjectivity: {result['subjectivity']}")
            st.write(f"Sentiment Score: {result['sentiment_score']}")

            st.subheader("Readability Metrics")
            st.write(f"Average Sentence Length: {result['Average Sentence Length']}")
            st.write(f"Percentage Complex Words: {result['Percentage Complex Words']}")
            st.write(f"Fog Index: {result['Fog Index']}")

            st.subheader("Word & Syllable Metrics")
            st.write(f"Total Words: {result['Total Words']}")
            st.write(f"Total Sentences: {result['Total Sentences']}")
            st.write(f"Average Words per Sentence: {result['Average Words per Sentence']}")
            st.write(f"Number of Complex Words: {result['Number of Complex Words']}")
            st.write(f"Cleaned Word Count: {result['Cleaned Word Count']}")
            st.write(f"Total Syllables: {result['Total Syllables']}")
            st.write(f"Average Syllables per Word: {result['Average Syllables per Word']}")
            st.write(f"Personal Pronouns: {result['Personal Pronouns']}")
            st.write(f"Average Word Length: {result['Average Word Length']}")
    else:
        st.error("No content found.")


elif uploaded_file is not None:
    urls_df = pd.read_csv(uploaded_file)
    url_col = None
    for col in urls_df.columns:
        if col.strip().lower() == "url":
            url_col = col
            break

    if url_col is None:
        st.error("No 'URL' column found in CSV.")
    else:
        urls = urls_df[url_col].dropna().tolist()
        st.sidebar.success(f"Processing {len(urls)} URLs...")

        results_list = []
        for link in urls:
            res = analyze_single_url(link)
            if res:
                results_list.append(res)

        if results_list:
            final_df = pd.DataFrame(results_list)
            st.subheader("Analysis Results (All URLs)")
            st.dataframe(final_df)

            
            csv_data = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="url_analysis_results.csv",
                mime="text/csv"
            )
        else:
            st.error("No results to display.")
