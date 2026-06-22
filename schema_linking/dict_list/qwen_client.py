"""Thin Qwen2.5 Coder 7B inference wrapper using vLLM."""

from __future__ import annotations

from typing import List, Optional

from zero_shot.prompts import extract_sql


def load_model(model_name: str, max_model_len: int, gpu_memory_utilization: float):
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for inference. Install it with: pip install vllm"
        ) from exc

    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    return llm, sampling_params


def call_qwen(prompt: str, llm=None, sampling_params=None, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct") -> str:
    if llm is None or sampling_params is None:
        llm, sampling_params = load_model(model_name, 4096, 0.7)

    tokenizer = llm.get_tokenizer()
    messages = [
        {"role": "system", "content": "You are a helpful SQL expert."},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate([formatted_prompt], sampling_params)
    raw_text = outputs[0].outputs[0].text
    return extract_sql(raw_text)


def batch_call_qwen(prompts: List[str], llm=None, sampling_params=None, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct") -> List[str]:
    if llm is None or sampling_params is None:
        llm, sampling_params = load_model(model_name, 4096, 0.7)

    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful SQL expert."},
            {"role": "user", "content": prompt},
        ]
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    outputs = llm.generate(formatted_prompts, sampling_params)
    return [extract_sql(output.outputs[0].text) for output in outputs]
