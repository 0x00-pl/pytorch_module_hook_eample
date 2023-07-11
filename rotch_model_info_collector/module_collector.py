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
        plt.savefig(os.path.join(output_dir, f'{output_name}.hist.png'))
        plt.clf()
        return num_small_value, num_total

    @staticmethod
    def plt_grid(data, output_name, output_dir='output'):
        plt.imshow(data.transpose(), cmap='viridis')
        plt.colorbar()  # 添加颜色条
        plt.title(output_name)
        plt.xticks(range(data.shape[-2]), [f'item{i}' for i in range(data.shape[-2])])
        # plt.show()
        plt.savefig(os.path.join(output_dir, f'{output_name}.grid.png'))
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
    print(f'overall attn sparsity is {collector.attn_sparsity[0] / collector.attn_sparsity[1] :.2f} == {collector.attn_sparsity}')
    pass


if __name__ == '__main__':
    main()
