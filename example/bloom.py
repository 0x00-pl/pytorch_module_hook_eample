from typing import Any, Tuple

import torch

from torch_model_info_collector import runtime, tools
from torch_model_info_collector.module_collector import ModuleCollector


class BloomCollector(ModuleCollector):
    def __init__(self):
        super().__init__()
        self.gelu_sparsity_threshold = 0.05
        self.gelu_sparsity = (0, 0)
        self.attn_sparsity_threshold = 0.05
        self.attn_sparsity = (0, 0)
        self.n_head = None

    def get_head_summary(self, tensors, n_head=32, name=''):
        norm_tensor = tools.torch_split_heads_and_normal(tensors[0], n_head)

        tensor_data_info = self.tensor_data_info(norm_tensor, name, self.attn_sparsity_threshold)
        self.plt_hist(
            norm_tensor, name,
            title=self.tensor_data_info(norm_tensor, name, self.attn_sparsity_threshold), output_dir='output/bloom'
        )
        self.attn_sparsity = self.update_attn_sparsity_from_tensor_data_info(self.attn_sparsity, tensor_data_info)

        self.plt_grid(norm_tensor[0].transpose(), name, output_dir='output/bloom')

    def get_gelu_summary(self, tensor: torch.Tensor, name=''):
        assert isinstance(tensor, torch.Tensor)
        tensor_data_info = self.tensor_data_info(tensor, name, self.gelu_sparsity_threshold)
        self.plt_hist(
            tensor, name, title=self.tensor_data_info(tensor, name,
                                                      self.gelu_sparsity_threshold), output_dir='output/bloom'
        )
        self.gelu_sparsity = self.update_attn_sparsity_from_tensor_data_info(self.gelu_sparsity, tensor_data_info)

    def get_hook(self, name: str):
        assert self.n_head is not None
        super_hook = super().get_hook(name)

        def hook(module: torch.nn.Module, inputs: Tuple[Any], outputs):
            super_hook(module, inputs, outputs)
            if name.endswith('self_attention.dense'):
                self.get_head_summary(inputs, self.n_head, name)
            elif name.endswith('mlp.gelu_impl'):
                self.get_gelu_summary(outputs, name)

        return hook

    def register_hook(self, model):
        self.n_head = model.config.n_head
        super().register_hook(model)


def main():
    collector = runtime.run_module('bigscience/bloom-7b1', BloomCollector())

    print(
        f'overall gelu sparsity is {collector.gelu_sparsity[0] / collector.gelu_sparsity[1] :.2f} '
        f'== {collector.gelu_sparsity}'
    )
    print(
        f'overall attn sparsity is {collector.attn_sparsity[0] / collector.attn_sparsity[1] :.2f} '
        f'== {collector.attn_sparsity}'
    )


if __name__ == '__main__':
    main()
