import pandas as pd
import numpy as np

def main():
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
    # merge the frames
    merged = pd.merge(credits, movies)
    print(merged)

    # crew and director sets are the same currently but that will change
    all_crews = merged[["title", "cast", "crew", "overview"]]
    all_directors = merged[["title", "cast", "crew", "overview"]]

    # divide crew
    crew_cutoff = np.random.rand(len(all_crews)) < 0.8
    crew_train = all_crews[crew_cutoff]
    crew_test = all_crews[~crew_cutoff]

    # divide director
    director_cutoff = np.random.rand(len(all_directors)) < 0.8
    director_train = all_directors[director_cutoff]
    director_test = all_directors[director_cutoff]

if __name__ == "__main__":
    main()