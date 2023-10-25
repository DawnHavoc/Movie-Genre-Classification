# Import necessary libraries
import pickle
from flask import Flask, request, render_template

import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name)

# Load the trained model from the pickle file
with open('D:/Projects/Movie_Genre_Classification/datasets/Genre Classification Dataset/linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

def process_plot(plot_description,column):   
    
   
    #  Text Lowercasing
    plot_description[column] = plot_description[column].str.lower()
    # Removing Special Characters and Punctuation
    plot_description[column] = plot_description[column].str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
    #  Remove html tags
    plot_description[column] = plot_description[column].apply(remove_html_tags)
    # Remove emoji
    plot_description[column] = plot_description[column].apply(remove_emojis)
    # Tokenization
    plot_description[column] = plot_description[column].apply(word_tokenize)
    #  Stop Word Removal
    stop_words = set(stopwords.words('english'))
    plot_description[column] = plot_description[column].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    # # Step 6: Spelling Correction (using TextBlob)
    # plot_description[column] = plot_description[column].apply(lambda tokens: " ".join([str(TextBlob(token).correct()) for token in tokens]))

    #  Stemming (using NLTK)
    stemmer = PorterStemmer()
    plot_description[column] = plot_description[column].apply(lambda tokens: " ".join([stemmer.stem(token) for token in tokens]))

    """
    cleaned_text = plot_description.lower()
    cleaned_text = re.sub(f"[{re.escape(string.punctuation)}]", "", cleaned_text)
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    blob = TextBlob(" ".join(filtered_tokens))
    corrected_text = str(blob.correct())
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in corrected_text]
    return " ".join(stemmed_tokens)
    """
    return plot_description


# Define a function to preprocess the input and make predictions
def predict_genre(plot_text):
    # Perform any necessary preprocessing on the plot text (e.g., text vectorization)
    # Replace this with your actual preprocessing steps
    # For simplicity, we're assuming the model accepts a numerical feature, so you might need NLP preprocessing here.
    plot_vector = ...  # Process the plot text into a numerical feature

    # Make a prediction using the loaded model
    predicted_genre = model.predict([plot_vector])
    
    # Return the predicted genre (you might have a mapping to convert numerical prediction to actual genre)
    return predicted_genre

# Define a route to handle the HTML form
@app.route('/', methods=['GET', 'POST'])
def predict_movie_genre():
    if request.method == 'POST':
        plot_text = request.form['plot']
        predicted_genre = predict_genre(plot_text)
        return render_template('result.html', predicted_genre=predicted_genre)
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
