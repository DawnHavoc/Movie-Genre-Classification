import os
import pandas as pd
import pathlib 

from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def lr_model(train_data,test_data,test_data_soln,X_train,X_test):
   
    
    # Train the Word2Vec model
    # model = Word2Vec(train_data['DESCRIPTION'], vector_size=100, window=5, min_count=1, sg=0)
        # Create and train the Logistic Regression model
    logistic_regression_model = LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(X_train,  train_data['GENRE-ENCODED'])

    # Predict the genres for the test data
    y_pred = logistic_regression_model.predict(X_test)

    # # Decode the label-encoded predictions
    # y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Calculate accuracy and display classification report
    accuracy = accuracy_score(test_data_soln['GENRE-ENCODED'], y_pred)
    # report = classification_report(test_data_soln['GENRE-ENCODED'], y_pred, target_names=label_encoder.classes_)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    # print(report)


def getfile():
    path=[]
    for dirname, _, filenames in os.walk('D:/Projects'): #'Projects' is the folder name in which the required files are saved
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

  
    
    # svm_model(pd.read_csv(train_set_file),pd.read_csv(test_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)
    # bert_model(pd.read_csv(train_set_file),pd.read_csv(test_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)
    lr_model(pd.read_csv(train_set_file),pd.read_csv(test_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)

if __name__ == "__main__":
    main()
