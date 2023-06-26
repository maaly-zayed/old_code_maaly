# import important libraries
import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.metrics import precision_score

def load_data(database_filepath):
    """
    The function to load the created database.
    Parameters:
    database_filepath: the file path to the database_filepath.
    """
    # create a connection to the sql database
    engine = create_engine('sqlite:///'+database_filepath)
    #read sql tables into datafram
    df = pd.read_sql_table('DisasterResponse', engine)
    # get the message data
    message = df['message']
    # get the categories name
    categories = df.iloc[:,4:]
    category_names = categories.columns 
    return message, categories, category_names

def tokenize(text):
    """
    The function to tokenize the text messages.
    Parameters:
    text: text messages.
    """
    # preprocessing using regex to detect url in the message
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # find all url in the message text
    detected_urls = re.findall(url_regex, text)
    # replace all url with "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

     # convert text messages into tokens
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # preprocessing
    lemm_lower_tokens = []
    for token in tokens:
        #lemmatize each token 
        process_token = lemmatizer.lemmatize(token).lower().strip()
        lemm_lower_tokens.append(process_token)
    return lemm_lower_tokens    



def build_model():
    
    """
    The function to create a pipeline for this model.
    Output: the function return the model
    """
    #create a pipline of CountVectorizer, TfidfTransformer and MultiOutputClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    #set the parameters 
    parameters = {
        'clf__estimator__n_estimators': [10],
    'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model





def evaluate_model(model, X_test, y_test, category_names):
    """
    The function to evaluate the model performance.
    Parameters:
    model: trained model.
    X_test: test text message
    y_test: the correct target
    category_names: the target names
    output: priniting the classification report
    """
    # get the prediction of test data
    y_pred = model.predict(X_test)
    for indx, category in enumerate(category_names):
        print(f'_____________{indx, category} ______________')
        print('________________________________________________')
        print(classification_report(list(y_test.values[:, indx]), list(y_pred[:, indx])))
        


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()