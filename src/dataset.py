import json
import os
import pickle
from collections import Counter
from typing import Dict
import torch
from transformers import AutoTokenizer
from utils import set_model_name

additional_prompts = {
    't5-small': {
        'inverse_dictionary': 'translate {} ',
        'hypernyms': 'What is a hypernym of "{}"?',
        'synonyms': 'What is a synonym of "{}"?',
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    't5-base': {
        'inverse_dictionary': 'What is the definiendum of "{}"?',
        'hypernyms': 'What is a hypernym of "{}"?',
        'synonyms': 'What is a synonym of "{}"?',
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    't5-large': {
        'inverse_dictionary': '"{}"?',
        'hypernyms': 'What is a hypernym of "{}"?',
        'synonyms': 'What is a synonym of "{}"?',
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    'GPT2': {
        'inverse_dictionary': '{} is called a "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    'GPT2-medium': {
        'inverse_dictionary': ' is called a "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
    'GPT2_large': {
        'inverse_dictionary': '{} is called a "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    'GPT2-large': {
        'inverse_dictionary': '{} is called a "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    'google_gemma_2b': {
        'inverse_dictionary': '{} is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    'gemma-2b': {
        'inverse_dictionary': '{} is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    'gemma_2b': {
        'inverse_dictionary': '{} is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
        'input_reconstruction': '{}',
        'exact_sequence_autoencoding': '{}',
    },
    'Meta-Llama-3-8B': {
        'inverse_dictionary': ' is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
    },
    'mistral-7b': {
        'inverse_dictionary': ' is called a ',  # ok sure
        'hypernyms': '{} is a type of',  # ok sure
        'synonyms': '{} is a synonym of',  # ok sure
    },
    'Llama-2-7b-chat-hf': {
        'inverse_dictionary': ' is called a ',  # ok sure
        'hypernyms': ' is a type of',  # ok sure
        'synonyms': ' is a synonym of',  # ok sure
    },
    'Meta-Llama-3-70B': {
        'inverse_dictionary': ' is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
    },
    'Llama-2-7b-hf': {
        'inverse_dictionary': ' is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
    },  #
    'Llama-2-7b': {
        'inverse_dictionary': ' is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
    },
    'Meta-Llama-3-8B-Instruct': {
        'inverse_dictionary': 'Find a term for the description {}. "',  # Accuracy: 1.61% (241/14991)
        'hypernyms': 'What is category or hypernym of {}? "',  # ok sure Accuracy: 0.64% (205/31962)
        'synonyms': 'Find synonyms for {}. "',  # ok sure - BEST
    },
    'mistral-7b-instruct': {
        'inverse_dictionary': 'What is {}? ',  # ok sure
        'hypernyms': 'Find a hypernym for {}. ',  # ok sure
        'synonyms': 'Find synonyms for {}. ',  # ok sure
    },
    'Llama-3.1-8B-Instruct': {
        'inverse_dictionary': 'Find a term for the description {}. "',  # Accuracy: 1.61% (241/14991)
        'hypernyms': 'What is category or hypernym of {}? "',  # ok sure Accuracy: 0.64% (205/31962
        'synonyms': 'Find synonyms for {}. "',  # ok sure - BEST
    },
    'meta_llama_Llama_3_2_1B': {  # 'Llama-3.2-3B':{
        'inverse_dictionary': '{} is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
    },
    'meta_llama_Llama_3_1_8B': {
        'inverse_dictionary': '{} is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
    },
    'meta_llama_Llama_3_2_3B': {
        'inverse_dictionary': '{} is called a "',  # ok sure
        'hypernyms': '{} is a type of "',  # ok sure
        'synonyms': '{} is a synonym of "',  # ok sure
    },
    'Qwen2.5-0.5B': {
        'inverse_dictionary': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
    'Qwen_Qwen2_5_0_5B': {
        'inverse_dictionary': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
    'qwen2.5-0.5b': {
        'inverse_dictionary': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
    'Qwen2.5-1.5B': {
        'inverse_dictionary': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
    'Qwen_Qwen2_5_1_5B': {
        'inverse_dictionary': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
    'Qwen2.5-3B': {
        'inverse_dictionary': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
    'Qwen_Qwen2_5_3B': {
        'inverse_dictionary': '{} is commonly known as "',
        'hypernyms': '{} is a type of "',
        'synonyms': '{} is a synonym of "',
    },
}


class Dataset:
    """
    Dataset class which reads the data from one of three variations:
    - Inverse dictionary (definition -> definiendum)
    - Synonyms
    - Hypernyms
    """
    def __init__(self, path: str, task_type: str, tokenizer=None, additional_prompt=None, model_name='GPT2',
                 load_from_cache=False, cache_file=None):
        self.path = path
        self.task_type = task_type
        self.model_name = model_name
        m_name = set_model_name(self.model_name)
        self.additional_prompt = additional_prompts[m_name][self.task_type]
        self.model_name = model_name

        if task_type not in ['inverse_dictionary', 'synonyms', 'hypernyms', 'input_reconstruction',
                             'exact_sequence_autoencoding']:
            raise ValueError(
                "data_type must be one of 'inverse_dictionary', 'synonyms', 'hypernyms' 'exact_sequence_autoencoding' or 'input_reconstruction'.")

        if tokenizer:
            self.tokenizer = tokenizer
        else:  # Default tokenizer GPT tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use the input path as the base for the cache file
        base_path, _ = os.path.splitext(self.path)
        if cache_file:
            self.cache_file = cache_file
        else:
            self.cache_file = f"{base_path}_{self.task_type}_{self.model_name}_cache.pkl"

        if load_from_cache and os.path.exists(self.cache_file):
            self.load(self.cache_file)
        else:
            self.data = self._load_data()
            self.end_before_padding = []
            # Tokenize the data and set the max length:
            # max length is either the max length of all data or what covers 95% of the data
            self.max_len = self._compute_max_length(ratio=0.95)
            self.tokenized_data = self.tokenize_data()
            self.word_idx = self._get_word_idx()

    def generate_prompt(self, item, masked=False):
        if self.task_type == 'inverse_dictionary':
            if self.additional_prompt:
                prompt = self.additional_prompt.format(item['definition'])
            else:
                prompt = f"{item['definition']}"
            definiendum = item['word']
        elif self.task_type == 'synonyms':
            prompt = self.additional_prompt.format(item['word']['word'])
            definiendum = item['synonym']['word']
        elif self.task_type == 'hypernyms':
            prompt = self.additional_prompt.format(item['word']['word'])
            definiendum = item['hypernym']['word']
        elif self.task_type == 'input_reconstruction':
            if masked:
                prompt = self.additional_prompt.format(item['corrupted_definition'])
            else:
                prompt = self.additional_prompt.format(item['definition'])
            definiendum = item['definition']
        elif self.task_type == 'exact_sequence_autoencoding':
            prompt = self.additional_prompt.format(item['definition'])
            definiendum = item['definition']

        return prompt, definiendum

    def _compute_max_length(self, ratio=0.95, masked=True) -> int:
        lengths = Counter()
        max_len = 0
        for item in self.data:
            prompt, _ = self.generate_prompt(item, masked)
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids
            lengths[tokenized_prompt.shape[1]] += 1
            max_len = max(max_len, tokenized_prompt.shape[1])
        if ratio == 100:  # Return the max length from all the data
            return max_len
        else:
            total = sum(lengths.values())
            current = 0
            for length, count in sorted(lengths.items()):  # Sort by length of prompt
                current += count
                if current / total >= ratio:
                    return length

    def _load_data(self) -> list:
        data = []
        with open(self.path, 'r') as file:
            dataset = json.load(file)
        if self.task_type == 'inverse_dictionary':
            data = [{'word': item['word'],
                     'definition': item['definition'],
                     'wordnet_id': item['wordnet_id']} for item in dataset]
        elif self.task_type == 'synonyms':
            for item in dataset:
                synonyms = item['synonyms']
                for synonym in synonyms:
                    data.append({'word': {'word': item['word'], 'wordnet_id': item['wordnet_id']}, 'synonym': synonym})
        elif self.task_type == 'hypernyms':
            for item in dataset:
                hypernyms = item['hypernyms']
                for hypernym in hypernyms:
                    data.append(
                        {'word': {'word': item['word'], 'wordnet_id': item['wordnet_id']}, 'hypernym': hypernym})

        elif self.task_type == 'input_reconstruction':
            for item in dataset:
                # print('item', item.keys())
                data.append({
                    'corrupted_definition': item['masked_definition'],
                    'definition': item['definition'],
                    'word': item['word'],
                    'wordnet_id': item['wordnet_id']
                })
        elif self.task_type == 'exact_sequence_autoencoding':
            for item in dataset:
                data.append({
                    # 'corrupted_definition': item['masked_definition'],
                    'definition': item['definition'],
                    'word': item['word'],
                    'wordnet_id': item['wordnet_id']
                })
        return data

    def tokenize_data(self):
        tokenized_data = []
        for item in self.data:
            if self.task_type == 'inverse_dictionary':
                prompt, definiendum = self.generate_prompt(item)
                tokenized_item = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True,
                                                max_length=self.max_len).input_ids
                tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding=False,
                                                       add_special_tokens=False).input_ids
                # tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding=False, add_special_tokens=False).input_ids
            elif self.task_type == 'synonyms':
                prompt, definiendum = self.generate_prompt(item)
                tokenized_item = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True,
                                                max_length=self.max_len).input_ids
                tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding=False,
                                                       add_special_tokens=False).input_ids
                # tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding=False, add_special_tokens=False).input_ids
            elif self.task_type == 'hypernyms':
                prompt, definiendum = self.generate_prompt(item)
                tokenized_item = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True,
                                                max_length=self.max_len).input_ids
                # tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding=False, add_special_tokens=False).input_ids
                tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding=False,
                                                       add_special_tokens=False).input_ids
            elif self.task_type == 'input_reconstruction':
                prompt, definiendum = self.generate_prompt(item, masked=True)
                tokenized_item = self.tokenizer(prompt, return_tensors='pt', truncation=True).input_ids
                tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding='max_length',
                                                       truncation=True, max_length=self.max_len).input_ids
            elif self.task_type == 'exact_sequence_autoencoding':
                prompt, definiendum = self.generate_prompt(item)
                tokenized_item = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True,
                                                max_length=self.max_len).input_ids
                print('tokenized_item', tokenized_item.shape)
                tokenized_definiendum = self.tokenizer(definiendum, return_tensors='pt', padding='max_length',
                                                       truncation=True,
                                                       max_length=self.max_len).input_ids
                print('tokenized_definiendum', tokenized_definiendum.shape)

            tokenized_data.append([tokenized_item, tokenized_definiendum])
        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return all
        text = self.data[idx]
        tokenized_text = self.tokenized_data[idx]
        word_idx = self.word_idx[idx]
        if 'corrupted_definition' in self.data[idx]:
            text['corrupted_definition'] = self.data[idx]['corrupted_definition']
        else:
            text['corrupted_definition'] = None
        return {'text': text, 'tokenized_text': tokenized_text, 'word_idx': word_idx}

    def __repr__(self):
        return f"Dataset with {len(self)} items and task type: {self.task_type}"

    def __str__(self):
        return f"Dataset with {len(self)} items and task type: {self.task_type}"

    def __delitem__(self, idx):
        if idx >= len(self) or idx < -len(self):
            raise IndexError("Index out of range")
        del self.data[idx]
        del self.tokenized_data[idx]
        del self.word_idx[idx]
        del self.end_before_padding[idx]

    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """Convert entire dataset to PyTorch tensors."""
        all_data = list(self)
        return {
            'input_ids': torch.cat([batch['input_ids'] for batch in all_data]),
            'labels': torch.cat([batch['labels'] for batch in all_data])
        }

    def _get_word_idx(self):
        pad_token_id = self.tokenizer.pad_token_id
        word_idxs = []

        for item in self.tokenized_data:
            word_idx = {"definition": [], "end_before_padding": []}
            # print(item[0].squeeze())
            input_ids = item[0].squeeze()
            word_idx["definition"] = 0
            attention_mask = (input_ids != pad_token_id).long()
            sequence_lengths = attention_mask.sum()  #
            if sequence_lengths >= self.max_len:
                word_idx["end_before_padding"] = self.max_len - 1
            else:
                word_idx["end_before_padding"] = sequence_lengths.item()
            self.end_before_padding.append(word_idx["end_before_padding"])
            word_idxs.append(word_idx)
        return word_idxs

    def save(self, file_path: str = None):
        """
        Save the processed dataset to a file.
        """
        if file_path is None:
            file_path = self.cache_file

        data_to_save = {
            'data': self.data,
            'end_before_padding': self.end_before_padding,
            'max_len': self.max_len,
            'tokenized_data': self.tokenized_data,
            'word_idx': self.word_idx,
            'task_type': self.task_type,
            'additional_prompt': self.additional_prompt,
            'model_name': self.model_name
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load(self, file_path: str = None):
        """
        Load a processed dataset from a file.
        """
        if file_path is None:
            file_path = self.cache_file

        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        self.data = loaded_data['data']
        self.end_before_padding = loaded_data['end_before_padding']
        self.max_len = loaded_data['max_len']
        self.tokenized_data = loaded_data['tokenized_data']
        self.word_idx = loaded_data['word_idx']
        self.task_type = loaded_data['task_type']
        self.additional_prompt = loaded_data['additional_prompt']
        self.model_name = loaded_data['model_name']
