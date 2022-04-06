import pickle
import pandas as pd
import sys

def main():
    input_file = sys.argv[1]
    # get text from input file
    with open(input_file, 'r') as f:
        overview = f.read()
    print(f"Overview: \n\n{overview}\n\n")

    title = ""
    cast = []

    # load pickle model
    director_model = pickle.load(open('director.pkl', 'rb'))
    director = director_model.predict([overview])[0]


    print(f"Title suggestion: {title}\n")
    print(f"Director suggestion: {director}\n")
    print(f"Cast suggestions: {cast}\n")

if __name__ == '__main__':
    main()