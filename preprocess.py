import pandas as pd
import numpy as np
import json
# nltk
import nltk
import pickle
# nltk.download('stopwords')
import ast


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
    # filter out all rows with less than 2 appearances
    atleast2 = df['crew'].value_counts()[df['crew'].value_counts() >= 2]
    df = df[df['crew'].isin(atleast2.index)]
    df.to_csv('df.csv')
    return df

def get_cast_df():
    df = pd.read_csv('df.csv')
    cast_df = pd.DataFrame(columns=['crew', 'cast', 'overview'])
    for i in range(len(df)):
        cast_members = ast.literal_eval(df['cast'].iloc[i])
        num_cast = 5
        for j in range(num_cast if len(cast_members) > 5 else len(cast_members)):
            cast_df = cast_df.append({'crew': df['crew'].iloc[i], 'cast': cast_members[j], 'overview': df['overview'].iloc[i]}, ignore_index=True)
    cast_df.to_csv('cast_df.csv')

def get_multi_label_df():
    cast_df = pd.read_csv('cast_df.csv')
    crew_df = pd.read_csv('df.csv')
    all_cast = cast_df['cast'].unique()
    df = pd.read_csv('df.csv')
    # keep overview and crew
    df = df[['crew', 'overview']]
    for cast in all_cast:
        df[cast] = 0
    for i in range(len(crew_df)):
        cast_members = ast.literal_eval(crew_df['cast'].iloc[i])
        num_members = 5 if len(cast_members) > 5 else len(cast_members)
        cast_members = cast_members[:num_members]
        for cast in cast_members:
            df.loc[df['crew'] == crew_df['crew'].iloc[i], cast] = 1
    df.to_csv('multi_label_df.csv')
            
def main():
    # df = get_df()
    # print(df.head())
    # get_multi_label_df()
    pass


if __name__ == "__main__":
    main()
