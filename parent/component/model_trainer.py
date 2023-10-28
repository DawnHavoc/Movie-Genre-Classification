import os
import pandas as pd
import pathlib 

from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import importlib.util
from tqdm import tqdm


# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def tune_model(train_data,X_train):
   
   
        # Define a range of hyperparameters to search
        param_grid = {
            # 'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
            # 'penalty': ['l1', 'l2'],  # Regularization type
            # 'solver': ['liblinear', 'saga']  # Solver algorithm
            'C': [0.01, 0.1, 1],  # Focus on these values
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # Create a Logistic Regression model
        model = LogisticRegression(max_iter=1000)

        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train,  train_data[source_file.LABEL_ENCODED_COLUMN])

        # Get the best hyperparameters
        return grid_search.best_params_, grid_search.best_estimator_

def train_model(best_model,test_data_soln,X_test):
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(test_data_soln[source_file.LABEL_ENCODED_COLUMN], y_pred)
    report = classification_report(test_data_soln[source_file.LABEL_ENCODED_COLUMN], y_pred)

    print("Classification Report of Logistic Regression:")
    print(report)
    print(f'Logistic Regression Accuracy: {accuracy}')



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
    X_train_tfidf = tfidf_vectorizer.fit_transform(pd.read_csv(train_set_file)[source_file.COLUMN_TO_CLEAN])
    X_test_tfidf = tfidf_vectorizer.transform(pd.read_csv(test_set_file)[source_file.COLUMN_TO_CLEAN])

    parameters,model=tune_model(pd.read_csv(train_set_file),X_train_tfidf)
    train_model(model,pd.read_csv(test_set_soln),X_test_tfidf)
if __name__ == "__main__":
    main()
