import json
import os
import argparse
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.dataset import Dataset
from src.utils import get_device, set_seed, set_model_name, load_model, tensor_to_list, predict_top_k_tokens, shift_right

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser(description='Test model predictions on original or fine-tuned models')
    parser.add_argument('--model_name', type=str, default='GPT2-large', help='Model name')
    parser.add_argument('--supervision_type', type=str, default='fine-tuned', help='(original or fine-tuned)')
    parser.add_argument('--seed', type=int, default=42, help='Random seeds')
    parser.add_argument('--task_type', type=str, default='inverse_dictionary', help='Task type (synonyms, hypernyms, inverse_dictionary, input_reconstruction, exact_sequence_autoencoding)')
    parser.add_argument('--model_path', type=str, default='models/GPT2_large/hypernyms_seed_42/checkpoint-1114/', help='Path to fine-tuned model')
    parser.add_argument('--k', type=int, default=1, help='Number of top-k predictions to consider')
    parser.add_argument('--data_path', type=str, default='data/wordnet_data_definitions.json', help='wordnet_data_hypernyms.json wordnet_data_definitions.json wordnet_data_synonyms.json Path to data file')
    parser.add_argument('--CARMA', default=False)
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    model = load_model(args, device, CARMA=False)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = 'right'
    model.eval()

    dataset = Dataset(args.data_path, task_type=args.task_type, tokenizer=model.tokenizer, model_name=args.model_name)
    args.model_name = set_model_name(args.model_name)

    if args.supervision_type == 'fine-tuned':
        file_name =f"{args.model_name}_correct_predictions_seed_{args.seed}_{args.task_type}_finetuned.json"
        _, temp_data = train_test_split(dataset, test_size=0.8, random_state=args.seed)
        _, dataset = train_test_split(temp_data, test_size=0.9, random_state=args.seed)
    else:
        file_name =f"{args.model_name}_correct_predictions_seed_{args.seed}_{args.task_type}.json"

    correct_predictions = 0
    total_predictions = 0
    correctly_predicted_data = []
    for i, sample in enumerate(tqdm(dataset, desc="Processing samples", unit="sample")):
        # if correct_predictions >= 2000:
        #     print("Reached the limit of 2000 predictions.")
        #     break
        if args.supervision_type == 'original':
            item = dataset.tokenized_data[i]
            prompt, label = item
        else:
            item = dataset[i]['tokenized_text']
            prompt, label = item

        if args.task_type in ['synonyms', 'hypernyms', 'inverse_dictionary']:
            if len(label[0]) != 1:
                continue
        total_predictions += 1
        if args.task_type in ['synonyms', 'hypernyms', 'inverse_dictionary']:
            label_id = label[0].item()
        elif args.task_type == 'input_reconstruction':
            label_ids = label[0]  # multi-token tensor

        if args.supervision_type == 'original':
            idx = dataset.word_idx[i]['end_before_padding']
        else:
            idx = dataset[i]['word_idx']['end_before_padding']
        if 't5' in args.model_name:
            decoder_input_ids = torch.tensor([[model.cfg.decoder_start_token_id]]).to(device)
        else:
            decoder_input_ids = torch.tensor([[model.tokenizer.bos_token_id]]).to(device)

        if args.task_type == 'input_reconstruction' or args.task_type == 'exact_sequence_autoencoding':
            if 't5' in args.model_name:
                # Shift the label (target) to create decoder input
                decoder_input_ids = shift_right(label[0].unsqueeze(0).to(device), model.tokenizer.pad_token_id)

                with torch.no_grad():
                    logits, _ = model.run_with_cache(
                        input=prompt.to(device),
                        decoder_input=decoder_input_ids
                    )
            else:
                # For GPT-like models, we can use the prompt directly
                logits, _ = model.run_with_cache(
                    input=prompt.to(device),
                )
            pred_ids = logits.argmax(dim=-1)

            predicted = model.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
            target = model.tokenizer.decode(label[0], skip_special_tokens=True).strip()

            if predicted == target:
                correct_predictions += 1
                correctly_predicted_data.append({
                    'text': sample['text'],
                    'tokenized_text': tensor_to_list(sample['tokenized_text']),
                    'word_idx': sample['word_idx'],
                })
                print(f"Item {i}: Correct reconstruction.\n→ Predicted: {predicted}\n→ Target:    {target} → Prompt: {model.tokenizer.decode(prompt[0])}")
        else:
            total_predictions += 1
            predictions = predict_top_k_tokens(prompt, idx, model, k=args.k, device=device)
            if label_id in predictions:
                correct_predictions += 1
                correctly_predicted_data.append({
                    'text': sample['text'],
                    'tokenized_text': tensor_to_list(sample['tokenized_text']),
                    'word_idx': sample['word_idx'],
                })
                print(f"Item {i}: Correct token prediction. Label: {model.tokenizer.decode([label_id])}, "
                      f"Rank: {predictions.index(label_id) + 1}")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nAccuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

    out_path = f"results/{args.model_name}"
    os.makedirs(out_path, exist_ok=True)
    with open(f"{out_path}/{file_name}", 'w') as f:
        json.dump(correctly_predicted_data, f, indent=2)


if __name__ == '__main__':
    main()
