# Import necessary libraries
import pickle
from flask import Flask, request, render_template


import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize


import importlib.util


source_file_path = "parent/constants/__init__.py"


spec = importlib.util.spec_from_file_location('__init__', source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

# Download the NLTK data needed for tokenization and stopwords
nltk.download(source_file.NLTK_DOWNLOAD)
nltk.download(source_file.NLTK_STOPWORDS)

app = Flask(__name__,static_folder='static', static_url_path='/static')

# Load the pickled model
with open(source_file.PRED_MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open(source_file.TFIDF_PATH, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Open the file in binary read mode and load the label encoder
with open(source_file.ENCODING_PATH, 'rb') as file:
    loaded_label_encoder = pickle.load(file)

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# remove emojis using a regular expression
def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # Emoticons
                           u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                           u"\U0001F700-\U0001F77F"  # Alphabetic presentation forms
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U0001F004-\U0001F0CF"  # Additional emoticons
                           u"\U0001F110-\U0001F251"  # Geometric Shapes Extended
                           u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
                           u"\U0001F910-\U0001F91E"  # Emoticons
                           u"\U0001F600-\U0001F64F"  # Emoticons
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def process_plot(sentence):   
    # Text Lowercasing
    sentence = sentence.lower()
    # Removing Special Characters and Punctuation
    sentence = re.sub(f"[{re.escape(string.punctuation)}]", "", sentence)
    # Remove html tags 
    sentence = remove_html_tags(sentence)
    # Remove emoji 
    sentence = remove_emojis(sentence)
    # Tokenization
    tokens = word_tokenize(sentence)
    # Stop Word Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming (using NLTK)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)


def predict_genre(plot_text):
  # Clean and preprocess the input plot
    cleaned_plot = process_plot(plot_text)
     # Use the pre-trained TF-IDF vectorizer to transform the input text
    input_tfidf = tfidf_vectorizer.transform([cleaned_plot])
    #make predictions 
    predicted_genre = model.predict(input_tfidf)
  
    #  use the loaded label encoder to decode labels
    decoded_genre = loaded_label_encoder.inverse_transform(predicted_genre)
    # convert it to a string and then strip the brackets
    decoded_genre_str = str(decoded_genre[0]).strip("[]")
  
    return decoded_genre_str

#  route to handle the HTML form
@app.route('/', methods=['GET', 'POST'])
def predict_movie_genre():
    if request.method == 'POST':
        plot_text = request.form['plot']
        predicted_genre = predict_genre(plot_text)
        return render_template('index.html', genres=predicted_genre)
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run()
