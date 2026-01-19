import argparse
import json
import logging
import os
from datetime import datetime

import torch
from jaxtyping import Float

from src.metrics import calculate_metrics, summary_metrics
from src.utils import get_device, set_seed, create_parts_ranges, set_model_name, create_groups_same_length, load_model, \
    shift_right, make_json_serializable, create_random_parts_ranges

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser(description='Activation Grouping')
    parser.add_argument('--model_name', type=str, default='GPT2', help='Model name')
    parser.add_argument('--supervision_type', type=str, default='original', help='Supervision type')
    parser.add_argument('--task_type', type=str, default='inverse_dictionary', help='Task type')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--component', type=str, default=None, help='Models components to group')
    parser.add_argument('--start_layer', type=int, default=4, help='Start layer')
    parser.add_argument('--norm_preserve', action='store_true', help='Preserve original norm after pooling')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--k', type=int, default=1, help='K')
    parser.add_argument('--grouping_protocol', type=str, default='mean', help='Grouping protocol')
    parser.add_argument('--granularity', type=str, default='token_to_words', help='Granularity')
    parser.add_argument('--broadcast_cap', action='store_true',
                        help='Broadcast pooled value back to original positions to preserve shape')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--CARMA', type=bool, default=False)
    args = parser.parse_args()

    # Set random seed and device
    set_seed(args.seed)
    device = get_device()

    # Assert the task type is valid
    assert args.task_type in ['inverse_dictionary', 'synonyms', 'hypernyms', 'input_reconstruction',
                              'exact_sequence_autoencoding'], 'Invalid task type'

    # Load the model based on supervision_type (fine-tuned or original)
    model = load_model(args)
    model.eval()

    # Set up logging
    args.model_name = set_model_name(args.model_name)
    log_path = f'results/{args.model_name}/{args.task_type}'
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{log_path}/llm_evaluation_results_{args.model_name}_{args.supervision_type}_{args.task_type}_broadcast_cap_{args.broadcast_cap}_norm_preserve_{args.norm_preserve}_seed_{args.seed}_{args.grouping_protocol}_{timestamp}_starting_{args.start_layer}.log'
    if args.granularity == 'token_to_phrases':
        granularity = 'tp_logs'
    elif args.granularity == 'token_to_words':
        granularity = 'tw_logs'
    elif args.granularity == 'random':
        granularity = 'ran_logs'
    log_file_json = f'{log_path}/{granularity}/llm_evaluation_results_{args.model_name}_{args.supervision_type}_{args.task_type}_broadcast_cap_{args.broadcast_cap}_norm_preserve_{args.norm_preserve}_seed_{args.seed}_{args.grouping_protocol}_{timestamp}_starting_{args.start_layer}.json'
    # os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_json), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print('Logging to:', log_file_json)

    # Load the dataset
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    max_length = len(dataset[0]['tokenized_text'][0][0])
    model_number_of_layers = model.cfg.n_layers
    result_collection = []

    # Prepare part ranges for the entire dataset
    part_ranges = []
    ratios = []

    for item in dataset:
        if args.granularity == 'token_to_phrases' or args.granularity == 'random':
            prompt = model.to_string(item["tokenized_text"][0][0][:item["word_idx"]["end_before_padding"]])
            tokenized_prompt = model.tokenizer.tokenize(prompt)
        elif args.granularity == 'token_to_words':
            prompt = model.to_string(item["tokenized_text"][0])
            tokenized_prompt = model.tokenizer.tokenize(prompt[0])
        if args.granularity == 'random':
            r, ratio_reduction = create_random_parts_ranges(
                tokenised_sentence=tokenized_prompt,
                end_before_padding=item["word_idx"]["end_before_padding"],
                max_length=max_length,
            )
        else:
            r, ratio_reduction = create_parts_ranges(
                tokenised_sentence=tokenized_prompt,
                end_before_padding=item["word_idx"]["end_before_padding"],
                max_length=max_length,
                granularity=args.granularity,
                model_name=args.model_name,
                sentence=prompt
            )
        part_ranges.append(r)
        ratios.append(ratio_reduction)

    grouped_batches = create_groups_same_length(part_ranges, args.batch_size)
    # print('grouped_batches', len(grouped_batches))
    print(grouped_batches.keys())

    # print('grouped_batches[2]', len(grouped_batches[2][0]), len(grouped_batches[2][0]))

    batches_ranges = {}
    ratio_ranges = {}
    for length, batch_groups in grouped_batches.items():
        batches_ranges[length] = []
        ratio_ranges[length] = []
        for batch in batch_groups:
            batches_ranges[length].append([part_ranges[idx] for idx in batch])
            ratio_ranges[length].append([ratios[idx] for idx in batch])

    # Define hook filters
    def create_hook_filter(components, layer_range, hook_types):
        return lambda name: name in components and any(
            [f".{i}." in name for i in range(layer_range[0], layer_range[1])]) and any(
            hook in name for hook in hook_types)

    # Process dataset in batches
    print('Starting the run ...')
    for length, batch_groups in grouped_batches.items():
        for batch_idx, batch in enumerate(batch_groups):
            clean_predictions, clean_labels, grouped_predictions, grouped_labels = [], [], [], []
            batch_inputs, batch_labels, batch_tokens = [], [], []

            # Collect tokenized inputs, labels, and their respective end token positions
            for idx in batch:
                tokenized_input = torch.tensor(dataset[idx]["tokenized_text"][0]).to(device)
                tokenized_label = torch.tensor(dataset[idx]["tokenized_text"][1]).to(device)

                batch_inputs.append(tokenized_input.squeeze())
                batch_labels.append(tokenized_label.squeeze())
                batch_tokens.append(dataset[idx]["word_idx"]["end_before_padding"])

            batch_inputs = torch.stack(batch_inputs)
            batch_labels = torch.stack(batch_labels)
            print('Batch inputs shape:', batch_inputs.shape)

            if 't5' in args.model_name:
                decoder_input_ids = shift_right(batch_labels[0].unsqueeze(0).to(device), model.tokenizer.pad_token_id)
                logits, cache = model.run_with_cache(input=batch_inputs.to(device),
                                                     decoder_input=decoder_input_ids
                                                     )
            else:
                # Run the model on the batch
                logits, cache = model.run_with_cache(batch_inputs)  # Clean run

            # Process clean predictions
            for i, token_end_idx in enumerate(batch_tokens):
                top_k_values, top_k_indices = torch.topk(logits[i, token_end_idx - 1, :], args.k)
                predicted_tokens_c = [model.to_string(top_k_indices[0])]
                if args.model_name in ["gemma-2b", "mistral-7b"]:
                    label_tokens = model.to_string(batch_labels[i][1:])
                else:
                    label_tokens = model.to_string(batch_labels[i])

                clean_predictions.append(predicted_tokens_c)
                clean_labels.append(label_tokens)

            # Define components and hook filters
            components = [name for name in cache.keys() if 'embed' not in name] if not args.component else [
                args.component]
            layer_range = (args.start_layer, model_number_of_layers)

            mlp_hook_filter = create_hook_filter(components, layer_range, ['mlp'])
            resid_hook_filter = create_hook_filter(components, layer_range, ['resid', 'hook_normalized', 'hook_scale'])
            attn_kqvz_hook_filter = create_hook_filter(components, layer_range,
                                                       ['hook_k', 'hook_q', 'hook_v', 'hook_z'])
            attn_scores_pattern_hook_filter = create_hook_filter(components, layer_range,
                                                                 ['hook_attn_scores', 'hook_pattern'])
            attn_out_hook_filter = create_hook_filter(components, layer_range, ['hook_attn_out'])
            final_hook_filter = create_hook_filter(components, layer_range, ['ln_final'])

            batch_range = batches_ranges[length][batch_idx]

            # Define grouping functions
            def group_mlp_resid_attn_out_layers2(tensor: Float, protocol: str = args.grouping_protocol,
                                                 ranges_r: list = batch_range, **kwargs):
                batch_size, sequence_length, hidden_size = tensor.shape
                device = tensor.device
                new_tensor = torch.zeros_like(tensor) if args.broadcast_cap else torch.zeros(
                    (batch_size, len(ranges_r[0]), hidden_size), device=device)

                for batch_idx in range(batch_size):
                    for i, (start, end) in enumerate(ranges_r[batch_idx]):
                        span = tensor[batch_idx, start:end]
                        if protocol == 'sum':
                            pooled = span.sum(dim=0)
                        elif protocol == 'mean':
                            pooled = span.mean(dim=0)
                        elif protocol == 'max':
                            pooled = span.max(dim=0).values
                        elif protocol == 'last_token':
                            # pooled = span[-1]
                            if span.shape[0] == 0:
                                continue  # or raise warning and skip
                            pooled = span[-1]

                        else:
                            raise ValueError(f"Invalid protocol: {protocol}")

                        if args.norm_preserve:
                            original_norm = span.norm(dim=-1).mean()
                            pooled_norm = pooled.norm()
                            scaling = (original_norm / pooled_norm).clamp(min=1e-6)
                            pooled = pooled * scaling

                        if args.broadcast_cap:
                            for j in range(start, end):
                                new_tensor[batch_idx, j] = pooled
                        else:
                            new_tensor[batch_idx, i] = pooled
                return new_tensor

            def group_mlp_resid_attn_out_layers(tensor: Float, protocol: str = args.grouping_protocol,
                                                ranges_r: list = batch_range, **kwargs):
                batch_size, sequence_length, hidden_size = tensor.shape
                device = tensor.device
                new_seq_len = len(ranges_r[0])
                if tensor.shape == (batch_size, new_seq_len, hidden_size):
                    return tensor
                new_tensor = torch.zeros((batch_size, new_seq_len, hidden_size), device=device)
                for batch_idx in range(batch_size):
                    for i, (start, end) in enumerate(ranges_r[batch_idx]):
                        span = tensor[batch_idx, start:end]  # [span_len, hidden_size]

                        if protocol == 'sum':
                            pooled = span.sum(dim=0)
                        elif protocol == 'mean':
                            pooled = span.mean(dim=0)
                        elif protocol == 'max':
                            pooled = span.max(dim=0).values
                        elif protocol == 'last_token':
                            # Take the last token in the range
                            if end > start:  # Ensure the range is valid
                                pooled = span[-1]  # equivalent to tensor[batch_idx, end - 1]
                            else:
                                # Handle edge case where range is invalid
                                pooled = torch.zeros(hidden_size, device=device)
                        else:
                            raise ValueError(f"Invalid protocol: {protocol}")

                        # Apply norm preservation if enabled
                        if args.norm_preserve and span.shape[0] > 0:
                            # Calculate original norm (average across the span)
                            original_norm = span.norm(dim=-1).mean()  # Average L2 norm across tokens
                            pooled_norm = pooled.norm()  # L2 norm of pooled vector

                            # Scale the pooled vector to match original norm
                            if pooled_norm > 1e-6:  # Avoid division by zero
                                scaling_factor = original_norm / pooled_norm
                                pooled = pooled * scaling_factor

                        new_tensor[batch_idx, i] = pooled
                return new_tensor

            def group_attn_scores_pattern_broadcast(tensor: Float, protocol: str = args.grouping_protocol,
                                                    ranges_r: list = batch_range, **kwargs):
                """
                Size-preserving broadcasting for attention scores/patterns.
                tensor: [B, H_a, K, K] -> returns same shape by filling each (span_i x span_j) block
                with a pooled per-head scalar.
                """
                B, H, K, _ = tensor.shape
                out = torch.zeros_like(tensor)

                for b in range(B):
                    r = ranges_r[b]
                    for i, (si, ei) in enumerate(r):
                        for j, (sj, ej) in enumerate(r):
                            block = tensor[b, :, si:ei, sj:ej]  # [H, len_i, len_j]

                            if block.numel() == 0:
                                continue

                            if protocol == 'sum':
                                pooled = block.sum(dim=(1, 2))  # [H]
                            elif protocol == 'mean':
                                pooled = block.mean(dim=(1, 2))  # [H]
                            elif protocol == 'max':
                                pooled = block.amax(dim=1).amax(dim=1)  # [H]
                            elif protocol == 'last_token':
                                pooled = tensor[b, :, ei - 1, ej - 1]  # [H]
                            else:
                                raise ValueError(f"Invalid protocol: {protocol}")

                            # (Optional) norm_preserve for attention is usually unnecessary; skip to avoid distortions but can be added if needed
                            out[b, :, si:ei, sj:ej] = pooled[:, None, None]  # broadcast into the block

                return out

            def group_qkvz_broadcast(tensor: Float, protocol: str = args.grouping_protocol,
                                     ranges_r: list = batch_range, **kwargs):
                """
                Size-preserving broadcasting for per-token multi-head tensors (Q/K/V/Z).
                tensor: [B, K, H_a, d_h] -> returns same shape by replacing tokens inside each span
                with a pooled per-head vector and broadcasting it across the span.
                """
                B, K, Hh, Dh = tensor.shape
                out = torch.zeros_like(tensor)

                for b in range(B):
                    r = ranges_r[b]
                    for (s, e) in r:
                        span = tensor[b, s:e]  # [len, Hh, Dh]
                        if span.numel() == 0:
                            continue

                        if protocol == 'sum':
                            pooled = span.sum(dim=0)  # [Hh, Dh]
                        elif protocol == 'mean':
                            pooled = span.mean(dim=0)  # [Hh, Dh]
                        elif protocol == 'max':
                            pooled = span.amax(dim=0)  # [Hh, Dh]
                        elif protocol == 'last_token':
                            pooled = span[-1]  # [Hh, Dh]
                        else:
                            raise ValueError(f"Invalid protocol: {protocol}")

                        if args.norm_preserve:
                            # Preserve mean L2 norm per head
                            orig = span.norm(dim=-1).mean(dim=0)  # [Hh]
                            new = pooled.norm(dim=-1)  # [Hh]
                            scale = torch.ones_like(new)
                            mask = new > 1e-6
                            scale[mask] = orig[mask] / new[mask]
                            pooled = pooled * scale[:, None]

                        # Broadcast pooled head vectors back to each token in the span
                        out[b, s:e] = pooled[None, :, :]
                return out + (tensor * (out == 0))

            def group_attn_scores_pattern(tensor: Float, protocol: str = args.grouping_protocol,
                                          ranges_r: list = batch_range, **kwargs):
                batch_size, num_heads, seq_len, _ = tensor.shape
                device = tensor.device
                new_seq_len = len(ranges_r[0])
                new_tensor = torch.zeros((batch_size, num_heads, new_seq_len, new_seq_len), device=device)
                if tensor.shape == (batch_size, num_heads, new_seq_len, new_seq_len):
                    return tensor
                for batch_idx, flattened_batch in enumerate(tensor):
                    r = ranges_r[batch_idx]
                    new_part = torch.zeros((num_heads, new_seq_len, new_seq_len), device=device)
                    for i, (start_i, end_i) in enumerate(r):
                        for j, (start_j, end_j) in enumerate(r):
                            # Extract the attention submatrix for this range pair
                            attn_block = flattened_batch[:, start_i:end_i, start_j:end_j]  # [num_heads, len_i, len_j]

                            if protocol == 'sum':
                                pooled = attn_block.sum(dim=1).sum(dim=1)  # [num_heads]
                            elif protocol == 'mean':
                                pooled = attn_block.mean(dim=1).mean(dim=1)  # [num_heads]
                            elif protocol == 'max':
                                pooled = attn_block.max(dim=1).values.max(dim=1).values  # [num_heads]
                            elif protocol == 'last_token':
                                # Take the attention from last token of range i to last token of range j
                                if end_i > start_i and end_j > start_j:  # Ensure ranges are valid
                                    pooled = flattened_batch[:, end_i - 1, end_j - 1]  # [num_heads]
                                else:
                                    # Handle edge case where ranges are invalid
                                    pooled = torch.zeros(num_heads, device=device)
                            else:
                                raise ValueError(f"Invalid protocol: {protocol}")

                            # Apply norm preservation if enabled
                            if args.norm_preserve and attn_block.numel() > 0:
                                # For attention scores, preserve the average magnitude
                                original_norm = attn_block.norm(dim=(1, 2))  # Norm across spatial dims for each head
                                original_norm = original_norm.mean()  # Average across heads
                                pooled_norm = pooled.norm()

                                if pooled_norm > 1e-6:  # Avoid division by zero
                                    scaling_factor = original_norm / pooled_norm
                                    pooled = pooled * scaling_factor

                            new_part[:, i, j] = pooled
                    new_tensor[batch_idx] = new_part
                return new_tensor

            def group_qkvz(tensor: Float, protocol: str = args.grouping_protocol, ranges_r: list = batch_range,
                           **kwargs):
                batch_size, seq_len, no_heads, hidden_size = tensor.shape
                device = tensor.device
                new_seq_len = len(ranges_r[0])
                if tensor.shape == (batch_size, new_seq_len, no_heads, hidden_size):
                    return tensor
                new_tensor = torch.zeros((batch_size, new_seq_len, no_heads, hidden_size), device=device)
                for batch_idx in range(batch_size):
                    for i, (start, end) in enumerate(ranges_r[batch_idx]):
                        span = tensor[batch_idx, start:end]  # [span_len, no_heads, hidden_size]

                        if protocol == 'sum':
                            pooled = span.sum(dim=0)  # [no_heads, hidden_size]
                        elif protocol == 'mean':
                            pooled = span.mean(dim=0)  # [no_heads, hidden_size]
                        elif protocol == 'max':
                            pooled = span.max(dim=0).values  # [no_heads, hidden_size]
                        elif protocol == 'last_token':
                            # Take the last token in the range
                            if end > start:  # Ensure the range is valid
                                pooled = span[-1]  # [no_heads, hidden_size]
                            else:
                                # Handle edge case where range is invalid
                                pooled = torch.zeros((no_heads, hidden_size), device=device)
                        else:
                            raise ValueError(f"Invalid protocol: {protocol}")

                        # Apply norm preservation if enabled
                        if args.norm_preserve and span.shape[0] > 0:
                            # For multi-head tensors, preserve norm per head
                            original_norm = span.norm(dim=-1).mean(dim=0)  # [no_heads] - average norm per head
                            pooled_norm = pooled.norm(dim=-1)  # [no_heads] - norm per head
                            # Avoid division by zero
                            valid_mask = pooled_norm > 1e-6
                            scaling_factor = torch.ones_like(pooled_norm)
                            scaling_factor[valid_mask] = original_norm[valid_mask] / pooled_norm[valid_mask]
                            # Apply scaling per head
                            pooled = pooled * scaling_factor.unsqueeze(-1)  # Broadcast across hidden_size dim

                        new_tensor[batch_idx, i] = pooled
                return new_tensor

            # Run model with hooks
            # Select grouping functions based on --broadcast_cap
            group_seq_fn = group_mlp_resid_attn_out_layers2 if args.broadcast_cap else group_mlp_resid_attn_out_layers
            group_attn_scores_fn = group_attn_scores_pattern_broadcast if args.broadcast_cap else group_attn_scores_pattern
            group_qkvz_fn = group_qkvz_broadcast if args.broadcast_cap else group_qkvz

            if 't5' in args.model_name:
                logits = model.run_with_hooks(
                    batch_inputs,
                    decoder_input=decoder_input_ids,
                    return_type='logits',
                    fwd_hooks=[  # fix this to do it over encoder~decoder
                        (mlp_hook_filter, group_mlp_resid_attn_out_layers),
                        (resid_hook_filter, group_mlp_resid_attn_out_layers),
                        (attn_scores_pattern_hook_filter, group_attn_scores_pattern),
                        (attn_kqvz_hook_filter, group_qkvz),
                        (attn_out_hook_filter, group_mlp_resid_attn_out_layers),
                        (final_hook_filter, group_mlp_resid_attn_out_layers)
                    ]
                )

            else:
                logits = model.run_with_hooks(
                    batch_inputs,
                    return_type='logits',
                    fwd_hooks=[
                        (mlp_hook_filter, group_seq_fn),
                        (resid_hook_filter, group_seq_fn),
                        (attn_scores_pattern_hook_filter, group_attn_scores_fn),
                        (attn_kqvz_hook_filter, group_qkvz_fn),
                        (attn_out_hook_filter, group_seq_fn),
                        (final_hook_filter, group_seq_fn),
                    ]
                )

            # Process grouped predictions
            for i, token_end_idx in enumerate(batch_tokens):
                top_k_values, top_k_indices = torch.topk(logits[i, -2, :],
                                                         args.k)  # -2 is the last token before padding -you could also use token_end_idx -1 adjusted for new length
                print('predicted top_k_indices:', top_k_indices)
                print('predicted top_k_values:', model.to_string(top_k_indices))
                predicted_tokens = [model.to_string(top_k_indices[0])]
                label_tokens = model.to_string(batch_labels[i])
                grouped_predictions.append(predicted_tokens)
                grouped_labels.append(label_tokens)
                print(
                    f"Batch {batch_idx}, Item {i}: Clean Prediction: {clean_predictions[i]}, Grouped Prediction: {grouped_predictions[i]}, Label: {label_tokens}")

            # Calculate metrics
            clean_metrics = calculate_metrics(clean_labels, clean_predictions, args.k)
            grouped_metrics = calculate_metrics(grouped_labels, grouped_predictions, args.k)

            print(f"Finished the grouping run for batch {batch_idx} of length {length}")

            # Prepare results
            results = {
                'model_name': args.model_name,
                'supervision_type': args.supervision_type,
                'task_type': args.task_type,
                'original_shape': max_length,
                'sequence_length': length,
                'granularity': args.granularity,
                'start_layer': args.start_layer,
                'component': args.component,
                'grouping_protocol': args.grouping_protocol,
                'grouped_shape': len(batch_range[0]),
                'seed': args.seed,
                'k': args.k,
                'clean_metrics': clean_metrics,
                'grouped_metrics': grouped_metrics,
                'grouped_predictions': grouped_predictions,
                'original_labels': grouped_labels,
                'batch_size': len(batch_inputs),
                'ratio_reduction': ratio_ranges[length][batch_idx],

            }
            result_collection.append(results)
            model.reset_hooks()
        print(f"Evaluation complete for length {length}. Results have been logged to '{log_file}'.")

    # Log the summary results
    results_summary = summary_metrics(result_collection)
    results_summary['granularity'] = args.granularity
    results_summary['start_layer'] = args.start_layer
    results_summary['component'] = args.component
    results_summary['grouping_protocol'] = args.grouping_protocol
    results_summary['seed'] = args.seed
    results_summary['k'] = args.k
    results_summary['model_name'] = args.model_name
    logging.info("Summary of results:")
    serializable_result_collection = [make_json_serializable(r) for r in result_collection]
    serializable_results_summary = make_json_serializable(results_summary)
    serializable_result_collection.append(serializable_results_summary)

    logging.info(json.dumps(serializable_result_collection, indent=2))
    os.makedirs(os.path.dirname(log_file_json), exist_ok=True)

    with open(log_file_json, 'w') as json_file:
        json.dump(serializable_result_collection, json_file, indent=4)
    print(f"All evaluations complete. Results logged to '{log_file}'")

    print(f"All evaluations complete. Results have been logged to '{log_file}'.")


if __name__ == '__main__':
    main()
