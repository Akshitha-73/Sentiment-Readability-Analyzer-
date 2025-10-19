import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize

# =====================
# Utility Functions
# =====================

def scrape_url(url):
    """Scrape the title and article content from a URL."""
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            title = soup.title.string.strip() if soup.title else None
            content = soup.find('article')
            content_text = content.get_text(separator=' ').strip() if content else None
            return title, content_text
        else:
            return None, f"Failed to fetch content. Status code: {res.status_code}"
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


def clean_text(text):
    """Lowercase and remove non-alphanumeric characters from text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_word_list(filepath):
    """Load a text file into a list of words."""
    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    return words


def filter_text(content, stopwords):
    """Tokenize content and remove stopwords."""
    words = word_tokenize(content)
    return [word for word in words if word.lower() not in stopwords]


def analyze_sentiment(words, positive_words, negative_words):
    """Analyze sentiment metrics from tokenized words."""
    positive_count = sum(1 for word in words if word.lower() in positive_words)
    negative_count = sum(1 for word in words if word.lower() in negative_words)

    # Avoid division by zero
    polarity = (positive_count - negative_count) / ((positive_count + negative_count) or 1)
    subjectivity = (positive_count + negative_count) / (len(words) or 1)
    sentiment_score = positive_count - negative_count

    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment_score": sentiment_score
    }


def process(df, stop_words, positive_words, negative_words):
    """Process a DataFrame and return sentiment analysis results."""
    results = []
    for _, row in df.iterrows():
        content = row.get("Cleaned Content")
        if pd.notnull(content):
            filtered_words = filter_text(content, stop_words)
            sentiment = analyze_sentiment(filtered_words, positive_words, negative_words)
            results.append({
                "positive_count": sentiment["positive_count"],
                "negative_count": sentiment["negative_count"],
                "polarity": sentiment["polarity"],
                "subjectivity": sentiment["subjectivity"],
                "sentiment_score": sentiment["sentiment_score"]
            })
    return results

def count_syllables(word):
    word = word.lower()
    vowels = "aeiou"
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    if word.endswith("e") and count > 1:
        count -= 1
    return count


def compute_readability_metrics(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    num_sentences = len(sentences)
    num_words = len(words)

    complex_words = [i for i in words if count_syllables(i) >= 3]
    num_complex_words = len(complex_words)

    average_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    percentage_complex_words = (num_complex_words / num_words) * 100 if num_words > 0 else 0

    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)


    return average_sentence_length, percentage_complex_words, fog_index

def calculate_metrics(text):
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)

    sentences = re.split(r'[.!?]+', text)
    total_sentences = len([s for s in sentences if s.strip()])

    average_words_per_sentence = total_words / total_sentences if total_sentences else 0

    return total_words, total_sentences, average_words_per_sentence

def complex_words(text):
    words = word_tokenize(text)
    complex_words = [w for w in words if count_syllables(w) >= 3]
    return len(complex_words)

def count_cleaned_words(text):
    words = word_tokenize(text)
    cleaned_words = [re.sub(r'[^\w\s]', '', word.lower()) for word in words if word.lower() not in stop_words]
    cleaned_words = [word for word in cleaned_words if word]
    return len(cleaned_words)

def count_syllables(word):
    word = word.lower()
    vowels = "aeiou"
    syllable_count = len(re.findall(r'[aeiou]+', word))
    if word.endswith("es") or word.endswith("ed"):
        if len(word) > 2 and not re.search(r'[aeiou]', word[:-2]):
            syllable_count -= 1
    return max(1, syllable_count)

def analyze_syllables(text):
    words = re.findall(r'\b\w+\b', text)  
    syllable_counts = [count_syllables(word) for word in words]
    return sum(syllable_counts), syllable_counts

def count_personal_pronouns(text):
    pronoun_pattern = r'\b(I|we|my|ours|us)\b'
    matches = re.findall(pronoun_pattern, text, flags=re.IGNORECASE)
    matches = [i for i in matches if i.lower() != "us"]
    return len(matches)

def calculate_average_word_length(text):
    words = re.findall(r'\b\w+\b', text)
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    if total_words == 0:
        return 0

    return total_characters / total_words