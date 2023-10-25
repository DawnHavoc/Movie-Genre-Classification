# import ast
# train_data['DESCRIPTION'] = train_data['DESCRIPTION'].apply(lambda x: ast.literal_eval(x))
# processed_data['TF-IDF'] = processed_data['TF-IDF'].apply(lambda x: np.array(eval(x)[0]))
import os
import pandas as pd
import pathlib 
import numpy as np


# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import torch
import tensorflow as tf
from transformers import  TFBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW


from sklearn.feature_extraction.text import TfidfVectorizer

def svm_model(train_data,test_data,test_data_soln,X_train,X_test):
        
    # ## Deserialize the 'TF-IDF' values from strings back to NumPy arrays
    # # train_data['TF-IDF'] = train_data['TF-IDF'].apply(lambda x: np.array([float(value) for value in x.split(";")]) if isinstance(x, str) else x)
   
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train,train_data['GENRE-ENCODED'] )
    predictions = svm_classifier.predict(X_test)

    # Create a DataFrame for the predictions
   
    test_data['GENRE'] = test_data_soln['GENRE']   # Add the predictions as a new column
    test_data['RESULT'] = predictions
    accuracy = accuracy_score(test_data_soln['GENRE-ENCODED'], predictions)
    print("Accuracy:", accuracy)

  
    # Save the results to a CSV file
    test_data.to_csv('D:/Projects/Movie-Genre-Classification/datasets/Genre Classification Dataset/result.csv', index=False)

# def bert_model(train_data,test_data,test_data_soln,X_train,X_test):
#     # Load a pre-trained BERT model and tokenizer
#     model_name = "bert-base-uncased"  # You can choose other BERT variants as needed
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

#     # Tokenize the input data
#     train_tokens = tokenizer(train_data['plot'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=256)
#     test_tokens = tokenizer(test_data['plot'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=256)

#     # Create PyTorch DataLoader for training and testing data
#     train_dataset = TensorDataset(train_tokens.input_ids, train_tokens.attention_mask, torch.tensor(train_data['GENRE-ENCODED'].tolist()))
#     train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     test_dataset = TensorDataset(test_tokens.input_ids, test_tokens.attention_mask, torch.tensor(test_data['GENRE-ENCODED'].tolist()))
#     test_dataloader = DataLoader(test_dataset, batch_size=8)

#     # Set up the optimizer
#     optimizer = AdamW(model.parameters(), lr=1e-5)

#     # Fine-tune the BERT model on your classification task
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     epochs = 3  # You can adjust the number of training epochs
#     for epoch in range(epochs):
#         model.train()
#         for batch in train_dataloader:
#             input_ids, attention_mask, labels = batch
#             input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()

#     # Evaluate the model
#     model.eval()
#     all_predictions = []
#     all_labels = []
#     for batch in test_dataloader:
#         input_ids, attention_mask, labels = batch
#         input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
#         with torch.no_grad():
#             outputs = model(input_ids, attention_mask=attention_mask)
#             logits = outputs.logits

#         predictions = np.argmax(logits.cpu().numpy(), axis=1)
#         all_predictions.extend(predictions)
#         all_labels.extend(labels.cpu().numpy())

#     # Decode the label-encoded predictions
#     y_pred_decoded = label_encoder.inverse_transform(all_predictions)

#     # Calculate accuracy and display classification report
#     accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
#     report = classification_report(all_labels, all_predictions, target_names=label_encoder.classes_)

#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     print(report)

def lr_model(train_data,test_data,test_data_soln,X_train,X_test):
   
    
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
    for filename in path:
        if(os.path.basename(filename)=='train_processed.csv'): #filename with extension
            train_set_filename=filename
        elif(os.path.basename(filename)=='test_processed.csv'): 
            test_set_filename=filename 
        elif(os.path.basename(filename)=='test_data_solution.csv'): 
            test_set_soln_filename=filename 
    return train_set_filename,test_set_filename,test_set_soln_filename

def main():
    train_set_file,test_set_file,test_set_soln=getfile()
      
      
   
    

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(pd.read_csv(train_set_file)['DESCRIPTION'])
    X_test_tfidf = tfidf_vectorizer.transform(pd.read_csv(test_set_file)['DESCRIPTION'])

    
    svm_model(pd.read_csv(train_set_file),pd.read_csv(test_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)
    # bert_model(pd.read_csv(train_set_file),pd.read_csv(test_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)
    # lr_model(pd.read_csv(train_set_file),pd.read_csv(test_set_file),pd.read_csv(test_set_soln),X_train_tfidf,X_test_tfidf)

if __name__ == "__main__":
    main()