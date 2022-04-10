import pickle

import nltk
import pandas as pd
import sys
from rake_nltk import Rake

import spacy
from nltk.corpus import wordnet as wn

def get_phrase(keywords, nouns):
    for phrase in keywords:
        words = nltk.word_tokenize(phrase)
        for word in words:
            if word in nouns:
                return phrase, word.lower()

def find_modifier(token):
    # iterate through the token's children
    if not token.children:
        return []
    for child in token.children:
        if child.dep_ == 'pobj':
            return convertNounToAdj(str(child))
        return find_modifier(child)



def create_title(phrase, noun, adjs, nlp):
    words = nltk.word_tokenize(phrase)
    title = [word for word in words if word.lower() == noun]
    orig_noun = title[0]
    phrase_len = len(words)
    processed = nlp(phrase)
    for index, token in enumerate(processed):
        if str(token).lower() in adjs and (index < phrase_len and words[index + 1] == orig_noun):
            title.insert(0, token)
        elif str(token).lower() == noun:
            mod = find_modifier(token)
            if mod:
                title.insert(0, mod[0][0])
    return " ".join(title)



def generate_title(description):
    # generates a title using RAKE keyword extraction and wordnet hacks
    nlp = spacy.load("en_core_web_sm")
    # nltk.download('omw-1.4')

    sents = nltk.sent_tokenize(description)
    tokenized = [[word for word in nltk.word_tokenize(sent)] for sent in nltk.sent_tokenize(description)]
    nouns = set()
    adjs = set()
    for sent in tokenized:
        tagged = nltk.pos_tag(sent)
        sent_nouns = [word.lower() for word, tag in tagged
                     if tag == 'NN' or tag == 'NNP'
                     or tag == 'NNS' or tag == 'NNPS']
        sent_adjs = [word.lower() for word, tag in tagged if tag == 'ADJ']
        nouns.update(sent_nouns)
        adjs.update(sent_adjs)
    r = Rake()
    r.extract_keywords_from_text(description)
    kw = r.get_ranked_phrases()
    top, noun = get_phrase(kw, nouns)
    title = create_title(top, noun, adjs, nlp)
    print(title)

def main():
    # example
    txt = """This is a no holds-barred thrilling drama mixed with killing, mayhem and manipulation among working 
    professionals. This film sheds light on a man's downfall from the pinnacles of success into the depths of his 
    damaged character. His insecurities lead him into a series of troubled romantic relationships and eventually a 
    web of events that include betrayal and murder."""
    generate_title(txt)

    # REMOVE BELOW LINE TO RUN REST OF PROG
    return
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


def convertNounToAdj(word):
    # uses wordnet (from nltk) to try and convert a noun to a synonymous adjective
    synsets = wn.synsets(word, pos='n')

    if not synsets:
        return []

    lemmas = [l for s in synsets
              for l in s.lemmas()
              if s.name().split('.')[1] in ('n')]

    related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    related_noun_lemmas = [l for drf in related_forms
                           for l in drf[1]
                           if l.synset().name().split('.')[1] in ('a',  's')]

    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    return result


if __name__ == '__main__':
    main()