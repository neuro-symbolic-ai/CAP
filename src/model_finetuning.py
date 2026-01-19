import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import argparse

from src.dataset import Dataset
from src.utils import set_seed, set_model_name

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NextTokenPredictionTrainer(TorchDataset):
    def __init__(self, tokenized_data, end_before_padding, max_len, tokenizer):
        self.tokenized_data = tokenized_data
        self.end_before_padding = end_before_padding
        self.max_len = max_len
        self.pad_token_id = tokenizer.pad_token_id  # Padding token ID

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        # Fetch tokenized input and the position of the end of the prompt
        tokens = self.tokenized_data[idx]
        prompt_end_idx = self.end_before_padding[idx]

        # Separate prompt and target word (assume that target follows the prompt)
        prompt = tokens[0][0][:prompt_end_idx]  # [prompt]
        target = tokens[1][0]  # [target]
        # input_ids = tokens['input_ids']  # [prompt + target]
        input_ids = torch.cat((prompt, target))
        # Apply truncation if input_ids exceed max_len
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        # Compute padding length after truncation
        padding_length = self.max_len - len(input_ids)

        # Pad input_ids if needed
        if padding_length > 0:
            input_ids = torch.cat((input_ids, torch.full((padding_length,), self.pad_token_id, dtype=torch.long)))

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        if padding_length > 0:
            attention_mask[-padding_length:] = 0  # Set the padding tokens in the mask to 0

        # Labels: copy input_ids, but mask the prompt (set to -100 for no loss on the prompt)
        labels = input_ids.clone()
        labels[:prompt_end_idx] = -100  # Mask prompt tokens
        if padding_length > 0:
            labels[-padding_length:] = -100  # Mask the padding in the labels

        return {
            'input_ids': input_ids,  # Prompt + target + padding
            'attention_mask': attention_mask,  # Mask real tokens vs padding
            'labels': labels  # Only compute loss on the target part, mask prompt and padding
        }


def main():
    parser = argparse.ArgumentParser('Model fine-tuning')
    parser.add_argument('--model_name', type=str, default='GPT2-large', help='Name of the model')
    parser.add_argument('--data_path', type=str, default='data/wordnet_data_definitions.json', help='Path to the data')
    parser.add_argument('--task_type', type=str, default='inverse_dictionary', help='Type of task')
    parser.add_argument('--seed', type=int, default=3, help='Random seed')
    parser.add_argument('--training_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--logging_steps', type=int, default=200, help='Logging steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save steps')
    parser.add_argument('--save_total_limit', type=int, default=1, help='Total save limit')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='Evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='Save strategy')
    parser.add_argument('--load_best_model_at_end', action='store_true', help='Load best model at end')
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss', help='Metric for best model')
    parser.add_argument('--greater_is_better', action='store_true', help='Whether greater metric is better')
    parser.add_argument('--report_to', type=str, default='none', help='Reporting destination')
    args = parser.parse_args()

    set_seed(args.seed)
    # device = get_device()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # model = model.to(device)
    model.eval()

    dataset = Dataset(args.data_path, task_type=args.task_type, tokenizer=tokenizer, model_name=args.model_name)
    args.model_name = set_model_name(args.model_name)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f'./models/{args.model_name}/{args.task_type}_seed_{args.seed}',  # Output directory
        num_train_epochs=args.training_epochs,  # Total number of training epochs
        per_device_train_batch_size=args.train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        # evaluation_strategy=args.evaluation_strategy,  # Evaluate at the end of each epoch
        save_strategy=args.save_strategy,  # Save checkpoints at the end of each epoch
        load_best_model_at_end=args.load_best_model_at_end,  # Load the best model at the end of training
        metric_for_best_model=args.metric_for_best_model,  # Choose a metric to monitor
        greater_is_better=args.greater_is_better,  # For loss, lower is better
        report_to=args.report_to  # Do not report to Hugging Face model hub
        )

    tokenized_data = dataset.tokenized_data
    end_before_padding = dataset.end_before_padding
    max_len = dataset.max_len

    # Split the data into train, validation, and test sets (e.g., 80/10/10 split)
    train_data, temp_data = train_test_split(list(zip(tokenized_data, end_before_padding)), test_size=0.8,
                                             random_state=args.seed)
    valid_data, test_data = train_test_split(temp_data, test_size=0.9,
                                             random_state=args.seed)  # Split the temp data into valid and test

    # Assuming tokenized_data and end_before_padding are precomputed from your Dataset class
    # Create datasets for train, validation, and test
    train_dataset = NextTokenPredictionTrainer(tokenized_data=[x[0] for x in train_data],
                                               end_before_padding=[x[1] for x in train_data],
                                               max_len=max_len,
                                               tokenizer=tokenizer)

    valid_dataset = NextTokenPredictionTrainer(tokenized_data=[x[0] for x in valid_data],
                                               end_before_padding=[x[1] for x in valid_data],
                                               max_len=max_len,
                                               tokenizer=tokenizer)

    test_dataset = NextTokenPredictionTrainer(tokenized_data=[x[0] for x in test_data],
                                              end_before_padding=[x[1] for x in test_data],
                                              max_len=max_len,
                                              tokenizer=tokenizer)
    print('size of train_dataset:', len(train_data))
    print('size of valid_dataset:', len(valid_data))
    print('size of test_dataset:', len(test_data))

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Use your custom dataset for training
        eval_dataset=valid_dataset  # Include validation dataset
    )

    # Start fine-tuning
    trainer.train()

    # Opt: Evaluate on the test dataset after training ~ I am not storing these results
    test_results = trainer.evaluate(test_dataset)
    print("Test results:", test_results)


if __name__ == '__main__':
    main()
