from transformers import AutoModelForCausalLM, AutoTokenizer


def run_module(model_name: str, collector=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if collector is not None:
        collector.register_hook(model)

    text = "This fruit shipping company provide different vehicle options like car and"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=1)

    output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, add_special_tokens=False)
    print(output_text)
    return collector
