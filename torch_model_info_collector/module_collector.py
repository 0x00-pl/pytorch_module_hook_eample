import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import torch


class ModuleCollector:
    def __init__(self):
        pass

    def tensor_info(self, tensor):
        if tensor is None:
            return None
        elif isinstance(tensor, torch.Tensor):
            return list(tensor.shape)
        elif isinstance(tensor, tuple):
            return tuple(self.tensor_info(i) for i in tensor)
        elif isinstance(tensor, dict):
            return {k: self.tensor_info(i) for k, i in tensor.items()}
        else:
            return '??'

    def get_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs: Tuple[Any], outputs):
            _ = module
            print(f'[hook]: {name} {self.tensor_info(inputs)} -> {self.tensor_info(outputs)}')

        return hook

    def register_hook(self, model):
        for name, module in model.named_modules():
            module: torch.nn.Module
            module.register_forward_hook(self.get_hook(name))

    @staticmethod
    def tensor_data_info(tensor, output_name, threshold):
        num_small_value = int(torch.sum(torch.lt(tensor, threshold)))
        num_total = tensor.numel()
        return {
            'count': num_small_value,
            'total': num_total,
            'fmt': f'{output_name} < {threshold}: '
                   f'{num_small_value} / {num_total} == {num_small_value / num_total :.2f}'
        }

    @staticmethod
    def update_attn_sparsity_from_tensor_data_info(sparsity, tensor_data_info):
        sparsity[0] += tensor_data_info['count']
        sparsity[1] += tensor_data_info['total']
        return sparsity

    @staticmethod
    def plt_hist(tensor: torch.Tensor, output_name, title=None, output_dir='output'):
        if title is None:
            title = output_name

        plt.title(title)
        plt.hist(tensor.flatten(), bins=100, log=True)
        # plt.ylim(0, 100)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{output_name}.hist.png'))
        plt.clf()

    @staticmethod
    def plt_grid(data, output_name, output_dir='output'):
        plt.imshow(data, cmap='viridis')
        plt.colorbar()  # 添加颜色条
        plt.title(output_name)
        # plt.xticks(range(data.shape[-2]), [f'item{i}' for i in range(data.shape[-2])])
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{output_name}.grid.png'))
        plt.clf()
