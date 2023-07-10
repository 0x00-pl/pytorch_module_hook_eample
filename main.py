import os
from collections import defaultdict
from typing import Tuple, Any

import numpy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomTokenizerFast, BloomForCausalLM
from transformers.models.bloom.modeling_bloom import BloomMLP, BloomAttention

import matplotlib.pyplot as plt


class LayerActiveCollector:
    def __init__(self):
        self.summary_dict = defaultdict(lambda: [])
        self.gelu_sparsity_threshold = 0.05
        self.gelu_sparsity = (0, 0)
        self.attn_sparsity_threshold = 0.05
        self.attn_sparsity = (0, 0)

    @staticmethod
    def get_head_summary(tensors, name=''):
        tensor = tensors[0]
        assert isinstance(tensor, torch.Tensor)
        n_head = 32
        head_tensor = torch.reshape(tensor, (*tensor.shape[:-1], n_head, -1))
        norm_tensor = head_tensor.norm(dim=-1)

        # debug
        plt.hist(tensor.flatten(), bins='auto')  # arguments are passed to np.histogram
        plt.savefig(f'output/{name}.hist.png')
        plt.clf()

        return norm_tensor

    def get_attention_summary(self, outputs, name=''):
        attention_output = outputs[0]
        assert isinstance(attention_output, torch.Tensor)
        norm_output = attention_output.norm(dim=-1)
        # debug
        tensor = norm_output
        plt.hist(tensor.flatten(), bins='auto')
        num_small_value = int((tensor.flatten() < self.attn_sparsity_threshold).sum())
        plt.title(
            f'{name} < {self.attn_sparsity_threshold}:  {num_small_value} / {tensor.numel()} == {num_small_value / tensor.numel() :.2f}'
        )
        plt.savefig(f'output/{name}.attn.png')
        plt.clf()
        self.attn_sparsity = (self.attn_sparsity[0] + num_small_value, self.attn_sparsity[1] + tensor.numel())
        return norm_output

    @staticmethod
    def get_mlp_summary(mlp_output, name=''):
        assert isinstance(mlp_output, torch.Tensor)
        norm_output = mlp_output.norm(dim=-1)
        return norm_output

    def get_gelu_summary(self, tensor: torch.Tensor, name=''):
        # debug
        plt.hist(tensor.flatten(), bins='auto')
        num_small_value = int((tensor.flatten() < self.gelu_sparsity_threshold).sum())
        plt.title(
            f'{name} < {self.gelu_sparsity_threshold}:  {num_small_value} / {tensor.numel()} == {num_small_value / tensor.numel() :.2f}'
        )
        plt.savefig(f'output/{name}.gelu.png')
        plt.clf()
        self.gelu_sparsity = (self.gelu_sparsity[0] + num_small_value, self.gelu_sparsity[1] + tensor.numel())

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
            print(f'[hook]: {name} {self.tensor_info(inputs)} -> {self.tensor_info(outputs)}')
            if isinstance(module, BloomAttention):
                self.summary_dict[name] += [self.get_attention_summary(outputs, name)]
            elif isinstance(module, BloomMLP):
                self.summary_dict[name] += [self.get_mlp_summary(outputs, name)]
            elif name.endswith('self_attention.dense'):
                self.summary_dict[name] += [self.get_head_summary(inputs, name)]
            elif name.endswith('mlp.gelu_impl'):
                self.summary_dict[name] += [self.get_gelu_summary(outputs, name)]

        return hook


def show_grid(data, output_name, output_dir='output'):
    plt.imshow(data.transpose(), cmap='viridis')
    plt.colorbar()  # 添加颜色条
    plt.xticks(range(data.shape[-2]), [f'item{i}' for i in range(data.shape[-2])])
    # plt.show()
    plt.savefig(os.path.join(output_dir, f'{output_name}.png'))
    plt.clf()


def main():
    tokenizer: BloomTokenizerFast = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    model: BloomForCausalLM = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", n_layer=30)  # n_layer=30
    collector = LayerActiveCollector()

    for name, module in model.named_modules():
        module: torch.nn.Module
        module.register_forward_hook(collector.get_hook(name))

    text = "This fruit shipping company provide different vehicle options like car and"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=1)

    output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, add_special_tokens=False)
    print(output_text)
    # print(collector.summary_dict.items())
    for layer_idx in range(model.config.n_layer):
        name = f'transformer.h.{layer_idx}.self_attention.dense'
        data = collector.summary_dict[name][0]
        show_grid(data[0].numpy(), name)

    print(f'overall gelu sparsity is {collector.gelu_sparsity[0] / collector.gelu_sparsity[1] :.2f} == {collector.gelu_sparsity}')
    pass


if __name__ == '__main__':
    main()
