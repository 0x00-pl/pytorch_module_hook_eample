from typing import Any, Tuple

import torch
from transformers import OPTForCausalLM

from torch_model_info_collector import runtime
from torch_model_info_collector.module_collector import ModuleCollector


class OptCollector(ModuleCollector):
    def __init__(self):
        super().__init__()
        self.gelu_sparsity_threshold = 0.05
        self.gelu_sparsity = (0, 0)
        self.attn_sparsity_threshold = 0.3
        self.attn_sparsity = (0, 0)
        self.num_attention_heads = None

    def get_head_summary(self, tensor, n_head=40, name=''):
        tensor = tensor.reshape((-1, tensor.shape[1], n_head, tensor.shape[2] // n_head))
        tensor_trans = tensor.transpose(-2, -3)
        norm_tensor = tensor_trans.norm(dim=-1)
        # self.plt_hist(tensor, self.attn_sparsity_threshold, name + '.raw', output_dir='output/opt')
        attn_sparsity = self.plt_hist(norm_tensor, self.attn_sparsity_threshold, name, output_dir='output/opt')
        self.attn_sparsity = tuple(sum(i) for i in zip(self.attn_sparsity, attn_sparsity))
        self.plt_grid(norm_tensor[0], name, output_dir='output/opt')

    def get_gelu_summary(self, tensor: torch.Tensor, name=''):
        assert isinstance(tensor, torch.Tensor)
        gelu_sparsity = self.plt_hist(tensor, self.gelu_sparsity_threshold, name, output_dir='output/opt')
        self.gelu_sparsity = tuple(sum(i) for i in zip(self.gelu_sparsity, gelu_sparsity))

    def get_hook(self, name: str):
        assert self.num_attention_heads is not None
        super_hook = super().get_hook(name)

        def hook(module: torch.nn.Module, inputs: Tuple[Any], outputs):
            super_hook(module, inputs, outputs)
            if name.endswith('.self_attn.out_proj'):
                self.get_head_summary(inputs[0], self.num_attention_heads, name)
            elif name.endswith('.self_attn.q_proj'):
                self.get_head_summary(outputs, self.num_attention_heads, name)
            elif name.endswith('.self_attn.k_proj'):
                self.get_head_summary(outputs, self.num_attention_heads, name)
            elif name.endswith('.self_attn.v_proj'):
                self.get_head_summary(outputs, self.num_attention_heads, name)
            elif name.endswith('.activation_fn'):
                self.get_gelu_summary(outputs, name)

        return hook

    def register_hook(self, model):
        self.num_attention_heads = model.config.num_attention_heads
        super().register_hook(model)


def main():
    collector = runtime.run_module('facebook/opt-13b', OptCollector())
    assert OPTForCausalLM

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
