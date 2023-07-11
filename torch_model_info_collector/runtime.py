from transformers import AutoModelForCausalLM, AutoTokenizer


def short_text():
    return "This fruit shipping company provide different vehicle options like car and"


def long_text():
    return '''
        Large language models (LLMs) with hundreds of
        billions of parameters have sparked a new wave
        of exciting AI applications. However, they are
        computationally expensive at inference time. Spar-
        sity is a natural approach to reduce this cost, but
        existing methods either require costly retraining,
        have to forgo LLM’s in-context learning ability, or
        do not yield wall-clock time speedup on modern
        hardware. We hypothesize that contextual sparsity,
        which are small, input-dependent sets of attention
        heads and MLP parameters that yield approxi-
        mately the same output as the dense model for a
        given input, can address these issues. We show that
        contextual sparsity exists, that it can be accurately
        predicted, and that we can exploit it to speed up
        LLM inference in wall-clock time without compro-
        mising LLM’s quality or in-context learning ability.
        Based on these insights, we propose DEJAVU, a
        system that uses a low-cost algorithm to predict
        contextual sparsity on the fly given inputs to each
        layer, along with an asynchronous and hardware-
        aware implementation that speeds up LLM
        inference. We validate that DEJAVU can reduce the
        inference latency of OPT-175B by over 2× com-
        pared to the state-of-the-art FasterTransformer,
        and over 6× compared to the widely used Hugging
        Face implementation, without compromising
        model quality.
    '''


def run_module(model_name: str, collector=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if collector is not None:
        collector.register_hook(model)

    text = short_text()
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, labels=inputs["input_ids"], max_length=1)

    output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, add_special_tokens=False)
    print(output_text)
    return collector
