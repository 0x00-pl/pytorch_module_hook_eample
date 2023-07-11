import torch


def torch_split_heads_and_normal(tensor, n_head):
    assert isinstance(tensor, torch.Tensor)
    head_tensor = torch.reshape(tensor, (*tensor.shape[:-1], n_head, -1))
    norm_tensor = head_tensor.norm(dim=-1)
    return norm_tensor
