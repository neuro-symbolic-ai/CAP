import random
import re
from typing import List, Tuple

import numpy as np
import torch
from nltk.tree import Tree
from transformer_lens import HookedTransformer, HookedEncoderDecoder
from transformers import AutoModelForCausalLM

from src.nlp_setup import nlp

domain_handling = {
    'photography': 'prefix', 'Hebrew': 'prefix', 'Ethiopia': 'prefix', 'slang': 'postfix',
    'genetics': 'prefix', 'cytology': 'prefix', 'Judaism': 'prefix', 'astronomy': 'prefix',
    'British': 'prefix', 'language': 'postfix', 'dentistry': 'prefix', 'anatomy': 'prefix',
    'Italian': 'prefix', 'computing': 'prefix', 'physics': 'prefix', 'philosophy': 'prefix',
    'Christianity': 'prefix', 'French': 'prefix', 'phonology': 'prefix', 'radiology': 'prefix',
    'theology': 'prefix', 'electricity': 'prefix', 'rhetoric': 'prefix', 'mineralogy': 'prefix',
    'neuroscience': 'prefix', 'psychoanalysis': 'prefix', 'antiquity': 'prefix', 'architecture': 'prefix',
    'printing': 'prefix', 'dressage': 'prefix', 'computer': 'prefix', 'biochemistry': 'prefix',
    'metaphor': 'postfix', 'geology': 'prefix', 'Yiddish': 'prefix', 'colloquial': 'postfix',
    'Zoroastrianism': 'prefix', 'agriculture': 'prefix', 'embryology': 'prefix', 'military': 'prefix',
    'economics': 'prefix', 'ballet': 'prefix', 'ethics': 'prefix', 'vertebrates': 'prefix',
    'chess': 'prefix', 'ecology': 'prefix', 'biology': 'prefix', 'cards': 'prefix',
    'archeology': 'prefix', 'England': 'prefix', 'neurophysiology': 'prefix', 'construction': 'prefix',
    'thermodynamics': 'prefix', 'mythology': 'prefix', 'Gnosticism': 'prefix', 'microbiology': 'prefix',
    'film': 'prefix', 'golf': 'prefix', 'physiology': 'prefix', 'craps': 'prefix',
    'bookkeeping': 'prefix', 'Britain': 'prefix', 'music': 'prefix', 'rugby': 'prefix',
    'microscopy': 'prefix', 'euphemism': 'postfix', 'furniture': 'prefix', 'cosmology': 'prefix',
    'voodooism': 'prefix', 'nautical': 'prefix', 'Hinduism': 'prefix', 'postpositive': 'postfix',
    'boxing': 'prefix', 'weapons': 'prefix', 'counterpoint': 'prefix', 'taxes': 'prefix',
    'baseball': 'prefix', 'accounting': 'prefix', 'fencing': 'prefix', 'pharmacology': 'prefix',
    'epidemiology': 'prefix', 'bowling': 'prefix', 'mechanics': 'prefix', 'color': 'postfix',
    'soccer': 'prefix', 'Apocrypha': 'prefix', 'geometry': 'prefix', 'statistics': 'prefix',
    'diplomacy': 'prefix', 'Luke': 'prefix', 'obstetrics': 'prefix', 'grammar': 'prefix',
    'zoology': 'prefix', 'trademark': 'postfix', 'politics': 'prefix', 'India': 'prefix',
    'chemistry': 'prefix', 'technology': 'prefix', 'horseshoes': 'prefix', 'Russia': 'prefix',
    'theater': 'prefix', 'logic': 'prefix', 'astrology': 'prefix', 'tennis': 'prefix',
    'immunology': 'prefix', 'games': 'prefix', 'poker': 'prefix', 'finance': 'prefix',
    'polo': 'prefix', 'linguistics': 'prefix', 'formerly': 'postfix', 'Brazil': 'prefix',
    'Latin': 'prefix', 'Marxism': 'prefix', 'metonymy': 'postfix', 'medicine': 'prefix',
    'religion': 'prefix', 'ophthalmology': 'prefix', 'phonetics': 'prefix', 'business': 'prefix',
    'meteorology': 'prefix', 'Haiti': 'prefix', 'psychology': 'prefix', 'mathematics': 'prefix',
    'obsolete': 'postfix', 'meat': 'prefix', 'poetic': 'postfix', 'verse': 'prefix',
    'game': 'prefix', 'folklore': 'prefix', 'angling': 'prefix', 'archaic': 'postfix',
    'formal': 'postfix', 'ornithology': 'prefix', 'Buddhism': 'prefix', 'Akkadian': 'prefix',
    'elections': 'prefix', 'bridge': 'prefix', 'spiritualism': 'prefix', 'paleontology': 'prefix',
    'corporation': 'prefix', 'prosody': 'prefix', 'pathology': 'prefix', 'historically': 'postfix',
    'broadcasting': 'prefix', 'sport': 'prefix', 'law': 'prefix', 'parliament': 'prefix',
    'figurative': 'postfix', 'falconry': 'prefix', 'sociology': 'prefix', 'forestry': 'prefix',
    'cricket': 'prefix', 'surgery': 'prefix', 'facetious': 'postfix', 'toxicology': 'prefix',
    'basketball': 'prefix', 'Briticism': 'postfix', 'orthopedics': 'prefix', 'pregnancy': 'prefix',
    'legend': 'prefix', 'sports': 'prefix', 'government': 'prefix', 'rare': 'postfix',
    'Fungi': 'prefix', 'psychiatry': 'prefix', 'botany': 'prefix', 'electronics': 'prefix',
    'neurology': 'prefix', 'football': 'prefix', 'optics': 'prefix', 'Islam': 'prefix',
    'plural': 'postfix', 'art': 'prefix', 'anthropology': 'prefix', 'metaphysics': 'prefix',
    'Bible': 'prefix', 'poetry': 'prefix', 'informal': 'postfix', 'histology': 'prefix',
    'mining': 'prefix', 'mountaineering': 'prefix', 'literary': 'postfix', 'heraldry': 'prefix'
}

delimiter_tokenisers = {'GPT2': 'Ġ', 'GPT2-medium': 'Ġ', 'GPT2-large': 'Ġ', 'GPT2_large': 'Ġ', 'GPT2-xl': 'Ġ',
                        'google_gemma_2b': '▁', 'gemma_2b': '▁', 'gemma-2b': '▁',
                        'Meta-Llama-3-8B-Instruct': 'Ġ', 'Meta-Llama-3-8B': 'Ġ', 'Meta-Llama-3-70B': 'Ġ',
                        'Meta-Llama-3-70B-Instruct': 'Ġ',
                        't5-small': '▁', 't5-base': '▁', 't5-large': '▁', 'Llama-3.1-8B-Instruct': 'Ġ',
                        'Llama-2-7b-chat-hf': '▁', 'mistral-7b': '▁', 'Llama-2-7b-hf': '▁', 'Llama-2-7b': '▁',
                        'Llama-3.2-3B': 'Ġ',
                        'meta_llama_Llama_3_2_3B': 'Ġ', 'meta_llama_Llama_3_2_1B': 'Ġ', 'qwen2_5_0_5b': 'Ġ',
                        'qwen2_5_1_5b': 'Ġ', 'qwen2_5_3b': 'Ġ', 'meta_llama_Llama_3_1_8B': 'Ġ',
                        'qwen2-0.5b': 'Ġ', 'Qwen2.5-0.5B': 'Ġ', 'Qwen2.5-1.5B': 'Ġ', 'Qwen2.5-3B': 'Ġ',
                        'Qwen_Qwen2_5_1_5B': 'Ġ', 'Qwen_Qwen2_5_3B': 'Ġ', 'Qwen_Qwen2_5_0_5B': 'Ġ', }

tokenized_first_word = {'GPT2': False, 'GPT2-medium': False, 'GPT2-large': False, 'GPT2_large': False,
                        'gemma-2b': False, 'gemma_2b': False,
                        'Llama-2-7b-chat-hf': True, 'Llama-2-7b-hf': True, 'Llama-2-7b': True, 'google_gemma_2b': False,
                        'Llama-3.1-8B-Instruct': False, 'Meta-Llama-3-70B-Instruct': False, 'Qwen2.5-1.5B': False,
                        'Qwen2.5-3B': False,
                        'Meta-Llama-3-8B': False, 'Meta-Llama-3-70B': False, 'Llama-3-8B-Instruct': False,
                        'meta_llama_Llama_3_1_8B': False, 'Qwen2.5-0.5B': False,
                        'qwen2-0.5b': False, 'qwen2_5_0_5b': False, 'qwen2_5_1_5b': False, 'qwen2_5_3b': False,
                        'Qwen_Qwen2_5_3B': False, 'Qwen_Qwen2_5_0_5B': False,
                        'Meta-Llama-3-8B-Instruct': False, 'meta_llama_Llama_3_2_3B': False,
                        'meta_llama_Llama_3_2_1B': False, 'Llama-3.2-3B': False, 't5-small': True, 't5-base': True,
                        't5-large': True,
                        'Qwen_Qwen2_5_1_5B': False}  # unless there is a space before it then this stands


# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)


def combine_wordnet_definitions(definition):
    parts = re.split(r'[;]+', definition)

    # Remove empty parts and parts starting with '-'
    parts = [p.strip() for p in parts if p.strip() and not p.strip().startswith('-')]

    # Keep the first part as the core, and process the rest
    core = parts.pop(0).strip() if parts else ""

    # Process remaining parts, ignoring any that contain 'followed by'
    parts = [p.strip() for p in parts if 'followed by' not in p]

    prefixes = [p for p in parts if any(d in p for d in domain_handling if domain_handling[d] == 'prefix')]
    postfixes = [p for p in parts if any(d in p for d in domain_handling if domain_handling[d] == 'postfix')]

    # Remaining parts that are not prefixes or postfixes
    others = [p for p in parts if p not in prefixes + postfixes]
    others_part = " and ".join(others).strip() if len(others) > 1 else " ".join(others).strip()

    prefix_part = f"in {', '.join(prefixes)}, " if prefixes else ""
    postfix_part = " ".join(postfixes).strip()

    combined_sentence = f"{prefix_part}{core}"
    if others_part:
        combined_sentence += f" and {others_part}"
    if postfix_part:
        combined_sentence += f" and {postfix_part}"

    return combined_sentence.strip()


def get_non_alphanumeric(s):
    return ''.join([char for char in str(s) if not char.isalnum() and not char.isspace()])


def get_device():
    # Check if CUDA is available or M1 chip
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.device('mps'):
        return 'mps'
    else:
        return 'cpu'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_constituent_labels(sentence, level=1):
    """
    Parse the given sentence and return constituent labels for each token.

    :param sentence: str, the input sentence to parse
    :param level: int, the number of levels to go up in the constituency tree (default: 1)
    :return: list of tuples, each containing (token, label)
    """

    def get_label_at_level(tree, token, current_level=0):
        if current_level == level:
            return tree.label()
        for subtree in tree:
            if isinstance(subtree, Tree):
                if token in subtree.leaves():
                    return get_label_at_level(subtree, token, current_level + 1)
        return 'O'  # Return 'O' if no label at the desired level is found ~ or we can return the token itself

    # Parse the sentence
    doc = nlp(sentence)

    # Get the constituency parse tree
    for sent in doc.sents:
        constituency_string = sent._.parse_string
        tree = Tree.fromstring(constituency_string)

        # Get labels for each token
        token_labels = []
        for token in sent:
            label = get_label_at_level(tree, token.text)
            token_labels.append((token.text, label))

        return token_labels, constituency_string


def clean_token(token):
    """
    Clean a token by removing special characters while preserving internal characters.

    :param token: str, the token to clean
    :return: str, the cleaned token
    """
    if token.startswith('Ġ') or token.startswith('▁'):
        return token[1:]
    return token


def align_huggingface_tokens_with_labels(sentence, hf_tokens, level=1):
    """
    Align Hugging Face tokenizer outputs with constituency parse labels.

    :param sentence: str, the input sentence
    :param hf_tokenizer: Hugging Face tokenizer instance
    :param level: int, the level of constituency parsing (default: 1)
    :return: list of tuples, each containing (subword_token, label)
    """
    # Get constituency labels
    word_labels, _ = get_constituent_labels(sentence, level)

    aligned_labels = []
    word_index = 0
    current_word = ""

    for hf_token in hf_tokens:
        clean_hf_token = clean_token(hf_token)

        if clean_hf_token:
            current_word += clean_hf_token

            # Check if we've completed a word
            while word_index < len(word_labels):
                word, label = word_labels[word_index]
                if word.lower().startswith(current_word.lower()):
                    aligned_labels.append((hf_token, label))
                    if current_word.lower() == word.lower():
                        word_index += 1
                        current_word = ""
                    break
                else:
                    word_index += 1
                    current_word = clean_hf_token
        else:
            # For special tokens or empty strings, use a special label
            aligned_labels.append((hf_token, 'SPECIAL'))

    return aligned_labels


def get_word_ranges_t5(tokens, return_ratio=True):
    # print('T5 tokenisation detected', tokens)

    word_ranges = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Skip specials
        if token == '</s>' or token.strip() == '':
            i += 1
            continue

        if token == '▁':
            # Append to next valid token
            j = i + 1
            while j < len(tokens) and tokens[j].strip() == '':
                j += 1
            if j < len(tokens):
                word_ranges.append((i, j + 1))
                i = j + 1
            else:
                break
        elif token.startswith('▁'):
            start = i
            i += 1
            while i < len(tokens) and not tokens[i].startswith('▁') and tokens[i] != '</s>':
                i += 1
            word_ranges.append((start, i))
        else:
            i += 1
    ratio_reduction = 0.0
    if return_ratio:
        # Count only non-special tokens
        effective_token_count = sum(
            1 for tok in tokens if tok not in {'</s>', '', '▁'} and not tok.isspace()
        )
        # Avoid divide-by-zero in edge cases
        if effective_token_count == 0:
            ratio_reduction = 0.0
        else:
            ratio_reduction = (effective_token_count - len(word_ranges)) / effective_token_count
    return word_ranges, ratio_reduction


def create_random_parts_ranges(tokenised_sentence, end_before_padding, max_length, min_chunk=1, max_chunk=4):
    """
    Randomly partitions a tokenized sentence into contiguous chunks.
    This simulates grouping without relying on linguistic structure.
    """
    ranges = []
    idx = 0
    while idx < end_before_padding:
        # Random chunk size between min_chunk and max_chunk, capped at remaining tokens
        chunk_size = random.randint(min_chunk, max_chunk)
        end_idx = min(idx + chunk_size, end_before_padding)
        ranges.append((idx, end_idx))
        idx = end_idx

    # Handle the padding segment (if any)
    if end_before_padding < max_length:
        ranges.append((end_before_padding, max_length))

    # Calculate reduction ratio
    org_length = end_before_padding + 1  # include padding
    new_length = len(ranges)
    ratio_of_reduction = (org_length - new_length) / org_length

    return ranges, ratio_of_reduction


def create_parts_ranges(tokenised_sentence, end_before_padding, max_length, granularity, model_name, sentence=None,
                        constituency_level=2):
    ranges = []
    if 't5' in model_name.lower():
        ranges, ratio_reduction = get_word_ranges_t5(tokenised_sentence)
        # print('Ranges:', ranges)
        # print('Ratio of reduction:', ratio_reduction)
        return ranges, ratio_reduction

    if granularity == 'token_to_words':
        current_word = ''
        current_word_start = 0
        current_word_end = 0
        delimiter = delimiter_tokenisers[model_name]
        flag_first_word = True
        i = 0
        while i < end_before_padding:
            # We check if we are at the start of a word and first word
            if flag_first_word:
                current_word = tokenised_sentence[i]
                current_word_start = i
                current_word_end = i
                flag_first_word = False
                i += 1
            else:
                # Check if we reached a new word
                if tokenised_sentence[i].startswith(delimiter):
                    current_word_end = i
                    if current_word_start > current_word_end:
                        print("Error: current_word_start > current_word_end")
                        print(f'Current word: {current_word}, start: {current_word_start}, end: {current_word_end}')
                    # print(f'Current word: {current_word}, start: {current_word_start}, end: {current_word_end}')
                    ranges.append((current_word_start, current_word_end))
                    current_word = tokenised_sentence[i]
                    current_word_start = i
                    current_word_end = i
                    i += 1
                else:
                    current_word += tokenised_sentence[i]
                    current_word_end = i
                    # print(f'Current word: {current_word}, start: {current_word_start}, end: {current_word_end}')
                    i += 1

        # Add the last word
        current_word_end += 1
        if current_word_start > current_word_end:
            print("Error: current_word_start > current_word_end")
            print(f'Last word: {current_word}, start: {current_word_start}, end: {current_word_end - 1}')
        # print(f'Last word: {current_word}, start: {current_word_start}, end: {current_word_end}')
        ranges.append((current_word_start, current_word_end))
        # The rest is the padding
        if current_word_end > max_length - 1:
            print("Error: current_word_start > current_word_end")
            print(f'Padding word: {current_word}, start: {current_word_start}, end: {max_length - 1}')
        # print(f'Last word: {current_word}, start: {end_before_padding}, end: {max_length-1}')
        ranges.append((end_before_padding, max_length))
    elif granularity == 'token_to_phrases':
        # We first annotate the sentences with the phrases (noun phrases, verb phrases, etc.)
        aligned_labels = align_huggingface_tokens_with_labels(sentence, tokenised_sentence, level=constituency_level)
        # print('Aligned Labels:', aligned_labels)
        current_label = ''
        current_start = 0
        current_end = 0
        for i, label in enumerate(aligned_labels):
            if i == 0:
                current_label = label[1]
                current_start = i
                current_end = i
            else:
                if label[1] == current_label:
                    current_end = i
                else:
                    ranges.append((current_start, current_end + 1))
                    current_label = label[1]
                    current_start = i
                    current_end = i
        # There is the last char and paddings
        ranges.append((current_start, current_end + 1))
        ranges.append((end_before_padding, max_length))
        # print('Ranges:', ranges)
    elif granularity == 'token_to_morphemes':  # don't
        pass
    elif granularity == 'token_to_pos':  # predicates and objects ~ not POS
        pass
    else:
        raise ValueError(f"Invalid granularity: {granularity}")
    org_length = len(tokenised_sentence[:end_before_padding]) + 1  # I will calculate the length of padding as one
    new_length = len(ranges)
    ratio_of_reduction = (org_length - new_length) / org_length
    return ranges, ratio_of_reduction


def create_groups_same_length(ranges):
    batches = {}
    for i, r in enumerate(ranges):
        number_hidden_states = len(r)
        if number_hidden_states not in batches:
            batches[number_hidden_states] = []
            batches[number_hidden_states].append(i)
        else:
            batches[number_hidden_states].append(i)
    for key, value in batches.items():
        print(f"Number of hidden states: {key}, Number of samples: {len(value)}")
    return batches


def split_into_batches(data, batch_size):
    values = []
    for i in range(0, len(data), batch_size):
        values.append(data[i:i + batch_size])
    return values


def create_groups_same_length(ranges, batch_size):
    batches = {}
    for i, r in enumerate(ranges):
        number_hidden_states = len(r)
        if number_hidden_states not in batches:
            batches[number_hidden_states] = []
        batches[number_hidden_states].append(i)
    for key, value in batches.items():
        batches[key] = split_into_batches(value, batch_size)
    return batches


def group_mlp_resid_attn_out_layers(tensor: torch.Tensor, ranges: List[List[Tuple[int, int]]],
                                    protocol: str = 'sum') -> torch.Tensor:
    '''
    Group the hidden states of the MLP layers according to the ranges (or residual stream, or attention stream output)
    :param tensor: the tensor of shape (batch_size, seq_len, hidden_size)
    :param ranges:  a list of lists of tuples, each list corresponds to a sample in batch, and each tuple corresponds to the start and end of the range
    :param protocol: how to group the hidden states, sum or mean or max
    :return: the new tensor of shape (batch_size, new_seq_len, hidden_size)
    '''
    batch_size, seq_len, hidden_size = tensor.shape
    device = tensor.device
    new_seq_len = len(ranges[0])
    new_tensor = torch.zeros((batch_size, new_seq_len, hidden_size), device=device)

    for batch_idx in range(batch_size):
        for i, value in enumerate(ranges[batch_idx]):
            start = value[0]
            end = value[1]
            if protocol == 'sum':
                new_tensor[batch_idx, i] = tensor[batch_idx, start:end].sum(dim=0)
            elif protocol == 'mean':
                new_tensor[batch_idx, i] = tensor[batch_idx, start:end].mean(dim=0)
            elif protocol == 'max':
                new_tensor[batch_idx, i] = tensor[batch_idx, start:end].max(dim=0).values
            else:
                raise ValueError(f"Invalid protocol: {protocol}")
    return new_tensor


def group_qkvz(tensor: torch.Tensor, ranges: List[List[Tuple[int, int]]], protocol: str = 'sum') -> torch.Tensor:
    device = tensor.device
    batch_size, seq_len, no_heads, hidden_size = tensor.shape
    new_seq_len = len(ranges[0])
    new_tensor = torch.zeros((batch_size, new_seq_len, no_heads, hidden_size), device=device)

    for batch_idx in range(batch_size):
        for i, (start, end) in enumerate(ranges[batch_idx]):
            if protocol == 'sum':
                new_tensor[batch_idx, i] = tensor[batch_idx, start:end].sum(dim=0)
            elif protocol == 'mean':
                new_tensor[batch_idx, i] = tensor[batch_idx, start:end].mean(dim=0)
            elif protocol == 'max':
                new_tensor[batch_idx, i] = tensor[batch_idx, start:end].max(dim=0).values
            else:
                raise ValueError(f"Invalid protocol: {protocol}")
    return new_tensor


def group_attn_scores_pattern(tensor: torch.Tensor, ranges: List[List[Tuple[int, int]]],
                              protocol: str = 'sum') -> torch.Tensor:
    '''
    original shape [batch_size, num_heads, seq_len, seq_len]
    new shape [batch_size, num_heads, new_seq_len, new_seq_len]
    :param tensor:
    :param ranges:
    :return:
    '''
    device = tensor.device
    batch_size, num_heads, seq_len, _ = tensor.shape
    new_seq_len = len(ranges[0])
    new_tensor = torch.zeros((batch_size, num_heads, new_seq_len, new_seq_len), device=device)

    for batch_idx, flattened_batch in enumerate(tensor):
        r = ranges[batch_idx]
        new_part = torch.zeros((num_heads, new_seq_len, new_seq_len), device=device)
        for i, (start, end) in enumerate(r):
            for j, (start_, end_) in enumerate(r):
                if protocol == 'sum':
                    new_part[:, i, j] = flattened_batch[:, start:end, start_:end_].sum(dim=1).sum(dim=1)
                elif protocol == 'mean':
                    new_part[:, i, j] = flattened_batch[:, start:end, start_:end_].mean(dim=1).mean(dim=1)
                elif protocol == 'max':
                    new_part[:, i, j] = flattened_batch[:, start:end, start_:end_].max(dim=1).values.max(dim=1).values
                else:
                    raise ValueError(f"Invalid protocol: {protocol}")
        new_tensor[batch_idx] = new_part

    # new_tensor = torch.stack(new_tensor)
    return new_tensor


def shift_right(input_ids, pad_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1]
    shifted[:, 0] = pad_token_id
    return shifted


def load_model(args, device=None, CARMA=False):
    """Load the model based on supervision type."""
    if args.supervision_type == 'fine-tuned':
        assert args.model_path is not None, "Model path must be provided for fine-tuned models"
        if CARMA:
            if 't5' in args.model_name:
                model = HookedEncoderDecoder.from_pretrained(args.model_name)
            else:
                model = HookedTransformer.from_pretrained(
                    args.model_name,
                    # center_unembed=args.model_name.startswith('GPT2'),
                    # center_writing_weights=args.model_name.startswith('GPT2'),
                    # fold_ln=args.model_name.startswith('GPT2'),
                    # refactor_factored_attn_matrices=args.model_name.startswith('GPT2'),
                )
            model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(args.model_path)
            if 't5' in args.model_name:
                model = HookedEncoderDecoder.from_pretrained(args.model_name)
            else:
                model = HookedTransformer.from_pretrained(
                    args.model_name,
                    hf_model=hf_model,
                    center_unembed=args.model_name.startswith('GPT2') or args.model_name.startswith('t5'),
                    center_writing_weights=args.model_name.startswith('GPT2') or args.model_name.startswith('t5'),
                    fold_ln=args.model_name.startswith('GPT2') or args.model_name.startswith('t5'),
                    refactor_factored_attn_matrices=args.model_name.startswith('GPT2') or args.model_name.startswith(
                        't5'),
                )
    elif args.supervision_type == 'original':
        if args.model_name.startswith('t5'):
            model = HookedEncoderDecoder.from_pretrained(args.model_name)
        else:
            model = HookedTransformer.from_pretrained(
                args.model_name,
                center_unembed=args.model_name.startswith('GPT2'),
                center_writing_weights=args.model_name.startswith('GPT2'),
                fold_ln=args.model_name.startswith('GPT2'),
                refactor_factored_attn_matrices=args.model_name.startswith('GPT2'),
                # n_devices=torch.cuda.device_count()
            )
    elif args.supervision_type == 'instruct':
        raise NotImplementedError("Instruction-based supervision is not yet implemented")
    return model


def set_model_name(model_name):
    model_name = model_name.replace(' ', '_').replace('-', '_').replace('/', '_')
    model_name = model_name.replace('.', '_').replace('(', '').replace(')', '')
    if model_name == "NousResearch/Llama-2-7b-chat-hf":
        return "Llama-2-7b-chat-hf"
    elif model_name == "meta-llama/Meta-Llama-3-8B":
        return "Meta-Llama-3-8B"
    elif model_name == "google/gemma-2b":
        return "gemma-2b"
    elif model_name == "mistralai/Mistral-7B-v0.1":
        return "mistral-7b"
    elif model_name == "mistralai/Mistral-7B-v0.3":
        return "mistral-7b"
    elif model_name == "meta-llama/Llama-2-7b-chat-hf":
        return "Llama-2-7b-chat-hf"
    elif model_name == "meta-llama/Llama-2-7b":
        return "Llama-2-7b"
    elif model_name == "meta-llama/Meta-Llama-3-70B":
        return "Meta-Llama-3-70B"
    elif model_name == "meta-llama/Llama-2-7b-chat-hf":
        return "Llama-2-7b-chat-hf"
    elif model_name == "meta-llama/Llama-2-7b-hf":
        return "Llama-2-7b"
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        return "Meta-Llama-3-8B-Instruct"
    elif '/' in model_name:
        return model_name.split('/')[-1]
    else:
        return model_name


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj


def predict_top_k_tokens(prompt, end_before_padding, model, k=1, device='cuda'):
    inputs = prompt.to(device)
    with torch.no_grad():
        logits = model(inputs)
        next_token_logits = logits[0, end_before_padding - 1, :]
        top_k_values, top_k_indices = torch.topk(next_token_logits, k)
        top_k_tokens = [idx.item() for idx in top_k_indices]
    return top_k_tokens


def make_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj
