# taken from https://github.com/morrisalp/fitw-wiki-data/blob/main/src/llm.py#L42

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch

class LlamaInference:

    def __init__(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",
        )

        self.tokenizer = tokenizer
        self.model = model
        self.pipe = pipe

    def _run_prompt(self, p, **kwargs):
        out = self.pipe(p, **kwargs)
        ans = out[0]['generated_text'][len(p):]
        ans = ans.strip()
        return ans
    
    def run_prompt(self, p, max_new_tokens=20):
        try:
            return self._run_prompt(p, max_new_tokens=max_new_tokens)
        except RuntimeError as e:
            print(f"RuntimeError while running prompt {p}")
            print(f"Error: {e}")
            return None