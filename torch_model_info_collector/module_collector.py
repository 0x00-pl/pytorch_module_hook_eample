import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import torch


class ModuleCollector:
    def __init__(self):
        pass

    def tensor_info(self, tensor):
        if isinstance(tensor, torch.Tensor):
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
    def plt_hist(tensor: torch.Tensor, threshold, output_name, output_dir='output'):
        plt.hist(tensor.flatten(), bins='auto')
        num_small_value = int(torch.sum(torch.lt(tensor, threshold)))
        num_total = tensor.numel()
        plt.title(
            f'{output_name} < {threshold}: '
            f'{num_small_value} / {num_total} == {num_small_value / num_total :.2f}'
        )
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{output_name}.hist.png'))
        plt.clf()
        return num_small_value, num_total

    @staticmethod
    def plt_grid(data, output_name, output_dir='output'):
        plt.imshow(data.transpose(-1, -2), cmap='viridis')
        plt.colorbar()  # 添加颜色条
        plt.title(output_name)
        # plt.xticks(range(data.shape[-2]), [f'item{i}' for i in range(data.shape[-2])])
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{output_name}.grid.png'))
        plt.clf()
