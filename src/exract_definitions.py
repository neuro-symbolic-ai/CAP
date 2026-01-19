import json
import random

from saf_datasets import WordNetFilteredDataSet
from src.utils import combine_wordnet_definitions


def corrupt_definition(defn, n_mask=2, mask="_"):
    """
    Randomly masks `n_mask` words in the input definition.
    """
    words = defn.split()
    if len(words) == 1:
        return None
    if len(words) <= n_mask:
        while len(words) <= n_mask:
            n_mask -=1
            if n_mask <= 0:
                return None
    mask_indices = random.sample(range(len(words)), k=n_mask)
    for i in mask_indices:
        words[i] = mask
    return " ".join(words)


def main(path, add_masked=True, n_mask=2, mask=" <extra_id_0> "):
    # Load the dataset
    dataset = WordNetFilteredDataSet()
    data_definitions = []
    # Extract the definitions
    for word in dataset:
        surface_str = str(word.surface)  # Ensure surface is treated as a string
        # surface_str = get_non_alphanumeric(surface_str) # non_alphanumeric_chars
        combined_definition = combine_wordnet_definitions(surface_str)
        if add_masked:
            # Add a masked version of the definition
            masked_definition = corrupt_definition(combined_definition, n_mask=n_mask, mask=mask)
            if masked_definition is None:
                continue
        data_definitions.append({
            'word': word.annotations['definiendum'],
            'definition': combined_definition,
            'wordnet_id': word.annotations['id'],
            'masked_definition': masked_definition if add_masked else None
        })

    # Save to JSON file
    with open(path+'wordnet_data_definitions.json', 'w') as f:
        json.dump(data_definitions, f, indent=4)


if __name__ == '__main__':
    main(path='data/')
