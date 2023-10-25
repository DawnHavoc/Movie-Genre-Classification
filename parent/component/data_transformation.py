import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder


import os
import pandas as pd
import pathlib 


# Download the NLTK data needed for tokenization and stopwords
nltk.download('punkt')
nltk.download('stopwords')

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Define a function to remove emojis using a regular expression
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

def process_genre(genre,column):
    label_encoded_column='GENRE-ENCODED'
     #  Text Lowercasing
    genre[column] = genre[column].str.lower()
    # Removing Special Characters and Punctuation
    genre[column] = genre[column].str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
    #  Remove html tags
    genre[column] = genre[column].apply(remove_html_tags)
    # Remove emoji
    genre[column] = genre[column].apply(remove_emojis)
    # label encoding the genre column
    label_encoder = LabelEncoder()
    genre[label_encoded_column] = label_encoder.fit_transform(genre[column])

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


def getfile():
    path=[]
    for dirname, _, filenames in os.walk('D:/Projects'): #'Projects' is the folder name in which the required files are saved
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    test_set_filename=""
    for filename in path:
        if(os.path.basename(filename)=='train_data.csv'): #filename with extension
            train_set_filename=filename
        elif(os.path.basename(filename)=='test_data.csv'): 
            test_set_filename=filename 
    return train_set_filename,test_set_filename

def batch_processing(data,filename):
  
    batch_size = 1000  
    column_to_clean = 'DESCRIPTION'
    column_to_encode='GENRE'
    processed_data=pd.DataFrame()
    
    if filename=='train_data':
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
        
            # Get the current batch of data
            batch_data = data.iloc[batch_start:batch_end]
            processed_data = pd.concat([processed_data, process_plot(batch_data,column_to_clean),process_genre(batch_data,column_to_encode)])

    elif filename=='test_data':
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
        
            # Get the current batch of data
            batch_data = data.iloc[batch_start:batch_end]
            processed_data = pd.concat([processed_data, process_plot(batch_data,column_to_clean)]) 
          
    
    return processed_data
    """
    Dealing with Short Words or Acronyms
    Custom handling based on your specific needs.

    Handling Negations
    Custom handling based on your specific needs.

    Remove Rare Words or Low-Frequency Terms
    Custom handling based on your specific needs.

    Addressing Data Imbalance
    Use resampling techniques like oversampling (e.g., SMOTE) or undersampling.
  
def tf_idf(train_data, test_data, column_to_clean, feature_name_column):
    # Create a TF-IDF vectorizer
    max_features = 1000  # You can adjust this value as needed
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    # Create empty lists to store TF-IDF matrices and feature names
    tfidf_strings_train = []  # Store TF-IDF values as strings for training data
    tfidf_strings_test = []   # Store TF-IDF values as strings for test data
    feature_names_list = []

    for dataset, tfidf_strings in [(train_data, tfidf_strings_train), (test_data, tfidf_strings_test)]:
        for row_index, row in dataset.iterrows():
            # Fit and transform the text data for the current row using TF-IDF
            tfidf_matrix = tfidf_vectorizer.fit_transform([row[column_to_clean]])

            # Convert the TF-IDF values to a string using a delimiter (e.g., semicolon)
            # tfidf_values = ";".join(map(str, tfidf_matrix.toarray().flatten()))
            tfidf_values =  tfidf_matrix

            # Get the feature names (unique words) from the vectorizer
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Append the TF-IDF values and feature names to the respective lists
            tfidf_strings.append(tfidf_values)
            feature_names_list.append(feature_names)

    # Create new columns for the TF-IDF values as strings in both training and test dataframes
    train_data['TF-IDF'] = tfidf_strings_train
    test_data['TF-IDF'] = tfidf_strings_test

    # Create a new DataFrame for the feature names
    feature_names_df = pd.DataFrame({feature_name_column: feature_names_list})


    return train_data, test_data
"""


def main():
    
  
    train_file='D:/Projects/Movie-Genre-Classification/datasets/Genre Classification Dataset/train_processed.csv'
    test_file='D:/Projects/Movie-Genre-Classification/datasets/Genre Classification Dataset/test_processed.csv'

    train_set_file,test_set_file=getfile()
    
    processed_train=batch_processing(pd.read_csv(train_set_file),os.path.splitext(os.path.basename(train_set_file))[0])
    processed_test=batch_processing(pd.read_csv(test_set_file),os.path.splitext(os.path.basename(test_set_file))[0])
    # train_tfidf,test_tfidf=tf_idf(processed_train,processed_test,column_to_clean,feature_name_column)
   
    
    processed_train.to_csv(train_file, index=False)
    processed_test.to_csv(test_file,index=False)


   
if __name__ == "__main__":
    main()

