from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import pandas as pd
import string
import os
#import pickle
#import joblib

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title='Home')

@app.route('/result')
def result():
    return render_template('result.html', title='Result')

@app.route('/result', methods=['POST'])
def predict():
    df = pd.read_csv('data/new_reviews.csv')
    df_data = df[['reviews', 'labels']]
    
    # Features and labels
    df_X = df_data['reviews']
    df_y = df_data['labels']

    # Define text processing function
    def text_process(tex):
        
        # Check characters to see if they are in punctuation
        no_punc = [char for char in tex if char not in string.punctuation]
        
        # Join the characters again to form the string.
        no_punc = ''.join(no_punc)
        
        # Remove stopwords and return cleaned text
        return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

    # Create Data pipeline 
    Log_reg_pipeline = Pipeline([('Log_reg_bow', CountVectorizer(analyzer=text_process,max_features=1000)),
                         ('Log_reg_tfidf', TfidfTransformer()),
                         ('Log_reg_classifier', LogisticRegression(class_weight='balanced'))])

    # Extract features
    corpus = df_X
    convertor = Log_reg_pipeline[:2]
    X = convertor.fit_transform(corpus) # Fit the Data

    # Split data into train and test sets
    rev_train, rev_test, label_train, label_test = train_test_split(X, df_y, test_size=0.3, random_state=42)


    # Modelling
    Log_reg_pipeline[2].fit(rev_train, label_train)
    Log_reg_pipeline[2].score(rev_test, label_test)


    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = convertor.transform(data).toarray()
        my_prediction = Log_reg_pipeline[2].predict(vect)
        # return 'lksjdf'

    return render_template('result.html', title='Result', prediction = my_prediction, comment=comment)



if __name__ == '__main__':
    app.run(debug=True)
