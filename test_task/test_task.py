# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:54:22 2018

@author: roman
"""

import click
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

@click.command()
@click.argument('phase')
@click.argument('drugs')


def main (drugs, phase):

    vectorizer = TfidfVectorizer(
         stop_words=ENGLISH_STOP_WORDS.union(frozenset(["plus", "+", "-", "|", ",",
                     "Drug", "Radiation", "Genetic", "Combination", "Product", "Biological", "Procedure", "Other",
                     "drug", "radiation", "genetic", "combination", "product", "biological", "procedure", "other"
                     ])))
    df = pd.read_csv("SearchResults.csv").dropna()
    df = df.filter(items=['NCT Number', 'Phases', 'Interventions'])
    df['Phases'] = df['Phases'].apply(lambda x: x.replace(' ', '_').replace('|', ' '))
    df['Interventions'] = df['Interventions'].apply(lambda x: x.replace('|', ' '))
    df['Phase_Interventions'] = df[['Phases', 'Interventions']].apply(lambda x: ' '.join(x), axis=1)
    X = vectorizer.fit_transform(df['Phase_Interventions'])
    n_clusters = 100
    kmeans = KMeans(n_clusters=n_clusters)
    df['Cluster'] = kmeans.fit_predict(X)
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    #for i in range(n_clusters):
        #print ("Cluster %d:" % i)
        #for ind in order_centroids[i, :10]:
            #print (' %s' % terms[ind])
        #print

    pred = predict(df, kmeans, phase, drugs, vectorizer)
    #print (df[df.Cluster == pred[0]])



def predict(df, kmeans, phase, drugs, vectorizer: TfidfVectorizer):

    phases = 'Phase_' + drugs
    #print (phases)
    drugs = phase.replace('|', ' ')
    #print('Drugs' + drugs)
    X = [phases + ' ' + drugs]
    X = vectorizer.transform(X)
    pred = kmeans.predict(X)

    print (df.loc[df.Cluster == pred[0], 'NCT Number'])

    return pred

if __name__ == "__main__":
    main()
































