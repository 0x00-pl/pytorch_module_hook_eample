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
        self.n_layers = None

    def get_head_summary(self, tensors, n_head=32, name=''):
        norm_tensor = tools.torch_split_heads_and_normal(tensors[0], n_head)
        attn_sparsity = self.plt_hist(norm_tensor, self.attn_sparsity_threshold, name, output_dir='output/bloom')
        self.attn_sparsity = (sum(i) for i in zip(self.attn_sparsity, attn_sparsity))
        self.plt_grid(norm_tensor, name, output_dir='output/bloom')

    def get_gelu_summary(self, tensor: torch.Tensor, name=''):
        assert isinstance(tensor, torch.Tensor)
        gelu_sparsity = self.plt_hist(tensor, self.gelu_sparsity_threshold, name, output_dir='output/bloom')
        self.gelu_sparsity = (sum(i) for i in zip(self.gelu_sparsity, gelu_sparsity))

    def get_hook(self, name: str):
        assert self.n_layers is not None
        super_hook = super().get_hook(name)

        def hook(module: torch.nn.Module, inputs: Tuple[Any], outputs):
            super_hook(module, inputs, outputs)
            if name.endswith('self_attention.dense'):
                self.get_head_summary(inputs, self.n_layers, name)
            elif name.endswith('mlp.gelu_impl'):
                self.get_gelu_summary(outputs, name)

        return hook

    def register_hook(self, model):
        self.n_layers = model.config.n_layers
        super().register_hook(model)


def main():
    model_name = 'facebook/opt-13b'
    collector = runtime.run_module(model_name, BloomCollector())

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
