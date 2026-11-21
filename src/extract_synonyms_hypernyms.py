"""Extract CAP's WordNet records with synset IDs.
"""

import json
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn


def download_wordnet():
    '''Download the WordNet corpus.'''
    try:
        wn.synsets("pen")
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            raise RuntimeError("Could not download the NLTK WordNet corpus")
        wn.synsets("pen")


def extract_synonyms(synset):
    '''Extract CAP's WordNet synsets.
    args: synset
    return: synsets'''
    base_word = synset.name().split(".")[0]
    return {
        "word": base_word,
        "wordnet_id": synset.name(),
        "synonyms": [
            {
                "word": lemma.name().replace("_", " "),
                # All lemmas in this record belong to the same synset.
                "wordnet_id": synset.name(),
            }
            for lemma in synset.lemmas()
            if lemma.name() != base_word
        ],
    }


def extract_hypernyms(synset):
    return {
        "word": synset.name().split(".")[0],
        "wordnet_id": synset.name(),
        "hypernyms": [
            {
                "word": hypernym.name().split(".")[0],
                "wordnet_id": hypernym.name(),
            }
            for hypernym in synset.hypernyms()
        ],
    }

def same_word(left, right):
    """Compare displayed words after normalising WordNet underscores and case."""
    def normalise(value):
        return " ".join(value.replace("_", " ").split()).casefold()

    return normalise(left) == normalise(right)

def main(path="data/", extra_filter=None):
    download_wordnet()
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_synonyms = []
    data_hypernyms = []
    for synset in wn.all_synsets():
        synonym_record = extract_synonyms(synset)
        if extra_filter:
            synonym_record["synonyms"] = [
                item for item in synonym_record["synonyms"]
                if not same_word(synonym_record["word"], item["word"])
            ]
        if synonym_record["synonyms"]:
            data_synonyms.append(synonym_record)

        hypernym_record = extract_hypernyms(synset)
        if extra_filter:
            hypernym_record["hypernyms"] = [
                item for item in hypernym_record["hypernyms"]
                if not same_word(hypernym_record["word"], item["word"])
            ]
        if hypernym_record["hypernyms"]:
            data_hypernyms.append(hypernym_record)

    for filename, records in (
        ("wordnet_data_synonyms.json", data_synonyms),
        ("wordnet_data_hypernyms.json", data_hypernyms),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            json.dump(records, stream, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
