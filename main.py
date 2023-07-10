from collections import defaultdict
from typing import Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomTokenizerFast, BloomForCausalLM
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomMLP, BloomAttention


class LayerActiveCollector:
    def __init__(self):
        self.summary_dict = defaultdict(lambda: torch.tensor(0))

    @staticmethod
    def get_attention_summary(output):
        assert isinstance(output, torch.Tensor)
        return torch.tensor(1)

    @staticmethod
    def get_mlp_summary(output):
        assert isinstance(output, torch.Tensor)
        return torch.tensor(1)

    def tensor_info(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return list(tensor.shape)
        elif isinstance(tensor, tuple):
            return tuple(self.tensor_info(i) for i in tensor)
        elif isinstance(tensor, dict):
            return {k: self.tensor_info(i) for k, i in tensor.items()}
        else:
            return '??'

    def get_hook(self, name):
        def hook(module: torch.nn.Module, inputs: Tuple[Any], outputs):
            print(f'[hook]: {name} {self.tensor_info(inputs)} -> {self.tensor_info(outputs)}')
            if isinstance(module, BloomAttention):
                self.summary_dict[name] += self.get_attention_summary(outputs)
            elif isinstance(module, BloomMLP):
                self.summary_dict[name] += self.get_mlp_summary(outputs)

        return hook


def main():
    tokenizer: BloomTokenizerFast = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    model: BloomForCausalLM = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", n_layer=27)  # n_layer=30
    collector = LayerActiveCollector()

    for name, module in model.named_modules():
        module: torch.nn.Module
        module.register_forward_hook(collector.get_hook(name))

    inputs = tokenizer("Hello, what is your name.", return_tensors="pt")
    outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=1)

    output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, add_special_tokens=False)
    print(output_text)
    print(collector.summary_dict.items())
    pass


if __name__ == '__main__':
    main()
