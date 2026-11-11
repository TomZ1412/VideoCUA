"""Response microbatches and right-padded loss targets for one prompt."""
import torch


def response_batches(groups, batch_size):
    if batch_size < 1:
        raise ValueError('grpo_response_batch_size must be positive')
    return [group[start:start + batch_size] for group in groups
            for start in range(0, len(group), batch_size)]


def batch_inputs(packages, item, shared_prefix):
    if not packages or any(p['index'] != packages[0]['index'] for p in packages):
        raise ValueError('A response microbatch must contain one prompt')
    width = max(max(len(p['ids']) for p in packages), 2 if shared_prefix else 1)
    ids = torch.zeros((len(packages), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    logps = torch.zeros_like(ids, dtype=torch.float32)
    for row, package in enumerate(packages):
        count = len(package['ids'])
        if count == 0 or count != len(package['logps']):
            raise ValueError('Each response needs nonempty IDs and matching sampling logprobs')
        ids[row, :count] = torch.tensor(package['ids'])
        mask[row, :count] = 1
        logps[row, :count] = torch.tensor(package['logps'])
    inputs = {k: v for k, v in item.items() if k in (
        'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'audio_feature') and v is not None}
    packed = ids.reshape(1, -1)
    inputs['input_ids'] = torch.cat((item['input_ids'], packed), dim=1)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    inputs['mm_token_type_ids'] = torch.cat((item['mm_token_type_ids'], torch.zeros_like(packed, dtype=torch.int32)), dim=1)
    if shared_prefix:
        inputs['grpo_shared_prefix'] = (item['input_ids'].shape[1], len(packages), width)
    elif len(packages) != 1:
        raise ValueError('Multiple responses require shared-prefix attention')
    inputs['grpo_targets'] = dict(
        selected_token_ids=ids, attention_mask=mask,
        advantages=torch.tensor([0.0 if p.get('padding') else p['advantage'] for p in packages]),
        sampling_per_token_logps=logps)
    return inputs
