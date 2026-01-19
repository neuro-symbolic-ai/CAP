import json
import os

import nltk
from nltk.corpus import wordnet as wn


def download_wordnet():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("WordNet not found. Downloading...")
        nltk.download('wordnet')


def extract_word_data(synset):
    word = synset.name().split('.')[0]
    synonyms = []
    for lemma in synset.lemmas():
        if lemma.name() != word:
            synonyms.append({
                'word': lemma.name().replace('_', ' '),
                'wordnet_id': f"{lemma.name()}.{synset.pos()}.{synset.name().split('.')[-1]}"
            })
    return {
        'word': word,
        'wordnet_id': synset.name(),
        'synonyms': synonyms
    }


def extract_synonyms(synset):
    base_word = synset.name().split('.')[0]
    synonyms = [
        {
            'word': lemma.name().replace('_', ' '),
            'wordnet_id': f"{lemma.name()}.{synset.pos()}.{synset.name().split('.')[-1]}"
        }
        for lemma in synset.lemmas()
        if lemma.name() != base_word
    ]
    return {
        'word': base_word,
        'wordnet_id': synset.name(),
        'synonyms': synonyms
    }


def extract_hypernyms(synset):
    word = synset.name().split('.')[0]
    hypernyms = []

    for hypernym in synset.hypernyms():
        hypernyms.append({
            'word': hypernym.name().split('.')[0],
            'wordnet_id': hypernym.name()
        })
    return {
        'word': word,
        'wordnet_id': synset.name(),
        'hypernyms': hypernyms
    }


def main(path, extra_filter=None):
    download_wordnet()
    data_synonyms = []
    data_hypernyms = []

    for synset in list(wn.all_synsets())[:]:
        word_data = extract_synonyms(synset)
        if word_data['synonyms']:  # Only add if there are synonyms
            data_synonyms.append(word_data)
        hypernym_data = extract_hypernyms(synset)
        if hypernym_data['hypernyms']:
            data_hypernyms.append(hypernym_data)

    if extra_filter:
        pass

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + 'wordnet_data_synonyms.json', 'w', encoding='utf-8') as f:
        json.dump(data_synonyms, f, indent=2, ensure_ascii=False)
    with open(path + 'wordnet_data_hypernyms.json', 'w', encoding='utf-8') as f:
        json.dump(data_hypernyms, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main(path='data/')