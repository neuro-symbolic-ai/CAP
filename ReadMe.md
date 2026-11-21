# Where Do LLMs Compose Meaning? A Layerwise Analysis of Compositional Robustness
[![Python 3.11.5](https://img.shields.io/badge/python-3.11.5-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


This repository contains the code for the paper "Where Do LLMs Compose Meaning? A Layerwise Analysis of Compositional Robustness" (EACL 2026).

## Overview

This project investigates how Large Language Models (LLMs) compose meaning across different layers by analysing their robustness to activation grouping at various granularities.

### Constituent-Aware Pooling (CAP)
We introduce **Constituent-Aware Pooling (CAP)**, a methodology grounded in compositionality, mechanistic interpretability, and information theory that intervenes in model activations by pooling token representations into linguistic constituents at various layers. The approach consolidates token-level activations into higher-level syntactic units (e.g., words, phrases) using different pooling strategies (mean, sum, max, last-token).


## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate [environment-name]

```
## Usage

## Usage

The main script `activation_grouping_main.py` supports various experimental configurations:

```bash
python activation_grouping_main.py \
    --model_name GPT2 \
    --task_type inverse_dictionary \
    --supervision_type original \
    --data_path path/to/data.json \
    --start_layer 4 \
    --grouping_protocol mean \
    --granularity token_to_words \
    --k 1 \
    --batch_size 16 \
    --seed 42 \
    --norm_preserve \
    --broadcast_cap
```

### Key Arguments

- `--model_name`: Model to evaluate (e.g., GPT2, gemma-2b)
- `--task_type`: Evaluation task (`inverse_dictionary`, `synonyms`, `hypernyms`, `input_reconstruction`, `exact_sequence_autoencoding`)
- `--supervision_type`: Use `original` or `fine-tuned` model
- `--granularity`: Grouping granularity (`token_to_words`, `token_to_phrases`, `random`)
- `--grouping_protocol`: Pooling method (`mean`, `sum`, `max`, `last_token`)
- `--start_layer`: Layer from which to start applying grouping
- `--norm_preserve`: Preserve activation norms after pooling (optional)
- `--broadcast_cap`: Broadcast pooled values back to original positions (optional)
- `--k`: Number of top-k predictions to consider


## Project Structure

```
.
├── activation_grouping_main.py    # Main evaluation script
├── src/
│   ├── metrics.py                 # Metric calculation utilities
│   └── utils.py                   # Helper functions
└── results/                       # Output logs and metrics
```

## Output

Results are saved in two formats:
- **Log files**: Detailed execution logs in `results/{model_name}/{task_type}/`
- **JSON files**: Structured metrics and predictions in `results/{model_name}/{task_type}/{granularity}_logs/`

Each result includes:
- Clean model predictions (no intervention)
- Grouped predictions (with activation pooling)
- Metrics comparing both conditions
- Compression ratios and layer information

## Experiments

The code supports analyzing:
- **Component-wise effects**: MLP, attention (Q/K/V/Z, scores, patterns), residual stream
- **Layer-wise progression**: Start interventions from different layers
- **Granularity levels**: Word-level, phrase-level, or random grouping
- **Pooling strategies**: Mean, sum, max, or last-token aggregation

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{aljaafari-etal-2026-llms,
    title = "Where Do {LLM}s Compose Meaning? A Layerwise Analysis of Compositional Robustness",
    author = "Aljaafari, Nura  and
      Carvalho, Danilo  and
      Freitas, Andre",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.214/",
    doi = "10.18653/v1/2026.eacl-long.214",
    pages = "4622--4646",
    ISBN = "979-8-89176-380-7",
    abstract = "Understanding how large language models (LLMs) process compositional linguistic structures is integral to enhancing their reliability and interpretability. We present Constituent-Aware Pooling (CAP), a methodology grounded in compositionality, mechanistic interpretability, and information theory that intervenes in model activations by pooling token representations into linguistic constituents at various layers. Experiments across eight models (124M-8B parameters) on inverse definition modelling, hypernym and synonym prediction reveal that semantic composition is not localised to specific layers but distributed across network depth. Performance degrades substantially under constituent-based pooling, particularly in early and middle layers, with larger models showing greater sensitivity. We propose an information-theoretic interpretation: transformers' training objectives incentivise deferred integration to maximise token-level throughput, resulting in fragmented rather than localised composition. These findings highlight fundamental architectural and training constraints requiring specialised approaches to encourage robust compositional processing."
}
```

## License

This project is licensed under the licensed under the GPLv3 License - see the [LICENSE](LICENSE.txt) file for details.

## Contact
For questions or issues regarding this code, or for paper-related inquiries, please:
- Open an issue in this repository
- Contact: nuraaljaafari@gmail.com