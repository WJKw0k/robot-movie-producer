import pandas as pd
import numpy as np
import json
# nltk
import nltk
# nltk.download('stopwords')


def get_cast_list(cast):
    cast_list = []
    for dict in cast:
        cast_list.append(dict['name'])
    return cast_list

def get_director(crew):
    for i in crew:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def preprocess_overview(overview):
    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    words = overview.lower().split()
    words = [w.strip(',') for w in words if w not in stopwords]
    # strip all non-alphanumeric characters from each word
    words = [''.join(c for c in w if c.isalnum()) for w in words]
    return ' '.join(words)

def get_df():
    credits = pd.read_csv('tmdb_5000_credits.csv')
    credits['cast'] = credits['cast'].apply(json.loads)
    credits['cast'] = credits['cast'].apply(get_cast_list)
    credits['crew'] = credits['crew'].apply(json.loads)
    credits['crew'] = credits['crew'].apply(get_director)

    movies = pd.read_csv('tmdb_5000_movies.csv')
   # merge on title
    df = pd.merge(movies, credits, on='title')
    # drop all columns except for id, title, and genres
    df = df[['title', 'crew', 'cast', 'overview']]
    # drop all rows with missing overview
    df = df.dropna(subset=['overview'])
    # preprocess overview
    df['overview'] = df['overview'].apply(preprocess_overview)
    # drop all rows with nan
    df = df.dropna()
    df.to_csv('df.csv')
    return df

def main():
    df = get_df()
    print(df.head())

    

if __name__ == "__main__":
    main()
