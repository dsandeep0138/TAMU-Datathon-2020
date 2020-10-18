from flask import Flask, render_template, request
import sys, ast, os, json, re, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy import spatial
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests,lxml
from selenium import webdriver

app = Flask(__name__)

#driver = webdriver.Chrome()
stop_words = set(stopwords.words('english'))
nltk.data.path.append('nltk_data')

def load_from_csv(filename):
    data_dict = {}
    
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return pd.Series(df.documents.values,index = df.Product_url).to_dict(), df
        
data_dict, modified_df = load_from_csv("product_urls_res.csv")

def data_processing(data):    #data is a list of lists
    res = []
    asciichars = set(string.printable)
    stemmer = PorterStemmer()
    for document in data.values():
        #print(document)
        document = ''.join(ch for ch in document if ch in asciichars)
        document = re.split("[" + string.punctuation + string.whitespace + "]+", document)
        document = [word for word in document if word.isalpha()]
        document = [word for word in document if len(word) > 1]
        document = [word for word in document if not word in stop_words]
        document = [stemmer.stem(word) for word in document]
        document = [word.lower() for word in document] 
        document = ' '.join(document)
        res.append(document)
                
    return res

res = data_processing(data_dict)

def generating_tf_idf(data,vocab = None):
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_df = 0.5,
                                 use_idf = True,
                                 ngram_range=(1, 2),vocabulary = vocab)
    vectors = vectorizer.fit_transform(data)
    
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense().tolist()

    return dense,feature_names


feature_df,vocab = generating_tf_idf(res)


def Reduce_Dim_SVD(feature_df,num_features = 100):
    svd = TruncatedSVD(num_features)
    lsa = make_pipeline(svd, Normalizer(copy = False))

    feature_data = lsa.fit_transform(feature_df)
    return feature_data, lsa

feature_data, lsa = Reduce_Dim_SVD(feature_df)

def VSM_Model(search_query,total_data,vocab, lsa):
    processed_query = data_processing({1:search_query})
    query_tf_idf,_ = generating_tf_idf(processed_query,vocab)
    query_reduced_dim = lsa.transform(query_tf_idf)
    
    scores_dict = []
    for data in total_data:
        scores_dict.append(1 - spatial.distance.cosine(data, query_reduced_dim[0]))
    return (np.argsort(scores_dict))[::-1]
    
@app.route("/", methods=['post', 'get'])
def home_page():
    query = ''
    if request.method == 'POST':
        query = request.form.get('query')

    #images = []
    if query == '':
        results = []
    else:
        search_results = VSM_Model(query, feature_data, vocab, lsa)
        modified_df["Features"] = feature_data.tolist()
        results = modified_df.iloc[search_results, :][:10]['Product_url'].tolist()

        '''
        for url in results:
            url = url.strip()
            url = url.strip("\"")
            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html)
            image = soup.find('img', itemprop = "image")
            images.append(image.get('src'))
        '''

    return render_template(
        'index.html',
        search_results = results)

if __name__ == "__main__":
    app.run(debug = True)
