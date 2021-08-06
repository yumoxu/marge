import torch

def avg_pool_token_reps(token_reps, token_masks):
    """
        avg pool token reps to sent reps.

    :param token_reps: d_batch * max_ns * max_nt * d_embed
    :param token_masks: d_batch * max_ns * max_nt
    :return:
        d_batch * max_ns * d_embed
    """
    sent_masks = torch.unsqueeze(token_masks, -2)  # d_batch * max_ns * 1 * max_nt
    nom = torch.matmul(sent_masks, token_reps)  # d_batch * max_ns * 1 * d_embed
    nom = torch.squeeze(nom, -2)  # d_batch * max_ns * d_embed

    # build denom factor
    n_words = torch.sum(token_masks, dim=-1, keepdim=True)  # d_batch * max_ns * 1
    n_words[n_words == float(0)] = 1  # for div, do not matter by replacing 0 with 1

    sent_embeds = nom / n_words

    return sent_embeds
