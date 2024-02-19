import torch.nn as nn


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model


def get_label(lang_x, tokenizer, mode='colon'):
    eoc_token = '<|endofchunk|>'
    media_token = '<image>'
    colon_token_id = tokenizer.encode(':')[0]
    eoc_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index(eoc_token)
    ]
    media_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index(media_token)
    ]
    label = lang_x.clone()
    # compute context len, by getting the index of the last colon token
    for idx in range(len(label)):
        if mode == 'colon':
            # get the last occurence of the ':' token
            # get a tensor of True/False values, then use torch.nonzero to get the indices
            indices = (label[idx] == colon_token_id).nonzero().flatten()
            # Then get the last occurrence
            end_of_context = indices[-1].item() + 1  # +1 because we want to include the colon token
        elif isinstance(mode, int):
            end_of_context = -label[idx].tolist()[::-1].index(media_token_id) - 1 + mode
        label[idx, : end_of_context] = -100
    label[label == tokenizer.pad_token_id] = -100
    label[:, 0] = -100
    label[label == media_token_id] = -100
    label[label == eoc_token_id] = -100
    return label