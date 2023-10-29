# import ast
# train_data['DESCRIPTION'] = train_data['DESCRIPTION'].apply(lambda x: ast.literal_eval(x))
# processed_data['TF-IDF'] = processed_data['TF-IDF'].apply(lambda x: np.array(eval(x)[0]))
import os
import pandas as pd
import pathlib 


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer

import importlib.util


# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)



def nb_model(train_data,test_data_soln,X_train,X_test):
    # Create and train the Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train, train_data[source_file.LABEL_ENCODED_COLUMN])

    # Make predictions on the test data
    predictions = naive_bayes_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(test_data_soln[source_file.LABEL_ENCODED_COLUMN], predictions)
   
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")

def rf_model(train_data,test_data_soln,X_train,X_test):
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  # You can adjust the number of estimators as needed

    # Fit the classifier to the training data
    rf_classifier.fit(X_train, train_data[source_file.LABEL_ENCODED_COLUMN])

    # Make predictions on the test data
    rf_predictions = rf_classifier.predict(X_test)

    # Calculate and print accuracy
    rf_accuracy = accuracy_score(test_data_soln[source_file.LABEL_ENCODED_COLUMN], rf_predictions)
   
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

def lr_model(train_data,test_data_soln,X_train,X_test):
   
    
    # Create and train the Logistic Regression model
    logistic_regression_model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000)
    logistic_regression_model.fit(X_train,  train_data[source_file.LABEL_ENCODED_COLUMN])

    # Predict the genres for the test data
    y_pred = logistic_regression_model.predict(X_test)



    # Calculate accuracy 
    accuracy = accuracy_score(test_data_soln[source_file.LABEL_ENCODED_COLUMN], y_pred)
   
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
   

def getfile():
    path=[]
    for dirname, _, filenames in os.walk(source_file.ROOT_DIR): #'Projects' is the folder name in which the required files are saved
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    test_set_filename=""
    test_set_soln_filename=""
    for filename in path:
        if(os.path.basename(filename)==source_file.TRAIN_SET_PROCESSED_NAME): #filename with extension
            train_set_filename=filename
        elif(os.path.basename(filename)==source_file.TEST_SET_PROCESSED_NAME): 
            test_set_filename=filename 
        elif(os.path.basename(filename)==source_file.TEST_SET_SOLN_PROCESSED_NAME): 
            test_set_soln_filename=filename 
    return train_set_filename,test_set_filename,test_set_soln_filename

def main():
    
    train_set_file,test_set_file,test_set_soln=getfile()
        
  
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(pd.read_csv(train_set_file)[source_file.COLUMN_TO_CLEAN])
    X_test_tfidf = tfidf_vectorizer.transform(pd.read_csv(test_set_file)[source_file.COLUMN_TO_CLEAN])

   
        
        
    rf_model(pd.read_csv(train_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)
    nb_model(pd.read_csv(train_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)
    lr_model(pd.read_csv(train_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)



if __name__ == "__main__":
    main()