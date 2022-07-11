import sys
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    Loads data and splits it into X and Y set

    Input: database filepath

    Output: X set, Y set, Category names
    '''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_df", con=engine)
    X = df["message"]
    Y = df.iloc[:, 4:]

    return X, Y, Y.columns


def tokenize(text):
    '''
    Tokenizing the text

    Input: Text string

    Output: Tokenized text
    '''
    
    tokens_var = word_tokenize(text)

    return tokens_var


def build_model():
    '''
    Build model's pipeline and runs GridSearchCV on max_depth

    Input: None

    Output: GridSearch pipeline object
    '''
    pipeline = Pipeline([
        ('BoW', CountVectorizer(tokenizer=tokenize)),
        ('Model_classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = { 'Model_classifier__estimator__max_depth':[3,5,7]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predicts result on test set and print classification report for each set

    Input: Model variable, X set, Y set, category names

    Output: None
    '''
    Y_pred = model.predict(X_test)
    for col in range(Y_pred.shape[1]):
        print(classification_report(Y_test.iloc[:, col], Y_pred[:, col],zero_division=0))
    


def save_model(model, model_filepath):
    '''
    Saves model into an pickle file
    
    Input: Model variable and destination filepath
    
    Output: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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