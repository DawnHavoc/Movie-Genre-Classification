import os
import pandas as pd
import pathlib 

from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

import importlib.util


# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def lr_model(train_data,test_data_soln,X_train,X_test):
   
    
        # Create and train the Logistic Regression model
    logistic_regression_model = LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(X_train,  train_data[source_file.LABEL_ENCODED_COLUMN])

    # Step 4: Save the trained model to a pickle file
    with open('D:/Projects/Movie-Genre-Classification/models/linear_regression_model.pkl', 'wb') as file:
        pickle.dump(logistic_regression_model, file)


def getfile():
    path=[]
    for dirname, _, filenames in os.walk('D:/Projects/Movie-Genre-Classification'): #'Projects' is the folder name in which the required files are saved
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    test_set_filename=""
    test_set_soln_filename=""
    for filename in path:
        if(os.path.basename(filename)=='train_processed.csv'): #filename with extension
            train_set_filename=filename
        elif(os.path.basename(filename)=='test_processed.csv'): 
            test_set_filename=filename 
        elif(os.path.basename(filename)=='test_data_solution_processed.csv'): 
            test_set_soln_filename=filename 
    return train_set_filename,test_set_filename,test_set_soln_filename

def main():
    train_set_file,test_set_file,test_set_soln=getfile()
      
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(pd.read_csv(train_set_file)['DESCRIPTION'])
    X_test_tfidf = tfidf_vectorizer.transform(pd.read_csv(test_set_file)['DESCRIPTION'])

    with open('D:/Projects/Movie-Genre-Classification/models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)

    
        # Initialize a label encoder
    label_encoder = LabelEncoder()

    # Fit the encoder and transform the labels
    train_labels = label_encoder.fit_transform(pd.read_csv(train_set_file)['GENRE'])
    test_labels = label_encoder.fit_transform(pd.read_csv(test_set_soln)['GENRE'])
    
    # Open the file in binary write mode and save the label encoder
    with open('D:/Projects/Movie-Genre-Classification/models/label_encoding.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    
    
    lr_model(pd.read_csv(train_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)

if __name__ == "__main__":
    main()
