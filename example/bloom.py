from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, BloomTokenizerFast

from rotch_model_info_collector.module_collector import ModuleCollector


class BloomCollector(ModuleCollector):
    def __init__(self):
        super().__init__()
        self.gelu_sparsity_threshold = 0.05
        self.gelu_sparsity = (0, 0)
        self.attn_sparsity_threshold = 0.05
        self.attn_sparsity = (0, 0)

    def get_head_summary(self, tensors, n_head=32, name=''):
        tensor = tensors[0]
        assert isinstance(tensor, torch.Tensor)
        head_tensor = torch.reshape(tensor, (*tensor.shape[:-1], n_head, -1))
        norm_tensor = head_tensor.norm(dim=-1)

        attn_sparsity = self.plt_hist(norm_tensor, self.attn_sparsity_threshold, name)
        self.attn_sparsity = (sum(i) for i in zip(self.attn_sparsity, attn_sparsity))

        return norm_tensor

    @staticmethod
    def get_mlp_summary(mlp_output, name=''):
        assert isinstance(mlp_output, torch.Tensor)
        norm_output = mlp_output.norm(dim=-1)
        return norm_output

    def get_gelu_summary(self, tensor: torch.Tensor, name=''):
        assert isinstance(tensor, torch.Tensor)
        gelu_sparsity = self.plt_hist(tensor, self.gelu_sparsity_threshold, name)
        self.gelu_sparsity = (sum(i) for i in zip(self.gelu_sparsity, gelu_sparsity))

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
        super_hook = super().get_hook(name)

        def hook(module: torch.nn.Module, inputs: Tuple[Any], outputs):
            super_hook(module, inputs, outputs)
            if name.endswith('self_attention.dense'):
                self.get_head_summary(inputs, 32, name)
            elif name.endswith('mlp.gelu_impl'):
                self.get_gelu_summary(outputs, name)

        return hook


def main():
    tokenizer: BloomTokenizerFast = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    model: BloomForCausalLM = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", n_layer=30)  # n_layer=30
    collector = BloomCollector()
    collector.register_hook(model)

    text = "This fruit shipping company provide different vehicle options like car and"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=1)

    output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, add_special_tokens=False)
    print(output_text)

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
