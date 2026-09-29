"""Optional Hugging Face adapter for decoder blocks used by Qwen/Mistral/Llama."""


def steering_hook(torch, delta):
    """Add the same calibrated update at the current final position on every forward."""
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(tensor):
            raise TypeError("Expected a decoder hidden-state tensor or tuple")
        adjusted = tensor.clone()
        adjusted[:, -1, :] += torch.as_tensor(delta, device=tensor.device, dtype=tensor.dtype)
        return (adjusted, *output[1:]) if isinstance(output, tuple) else adjusted
    return hook


class LocalBackend:
    temperature = 0.7

    def __init__(self, model_id, layer=14):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Install optional dependencies with: pip install -e ".[local]"') from exc
        self.torch = torch
        self.model = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.network = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=False).to(device).eval()
        blocks = getattr(getattr(self.network, "model", None), "layers", None)
        if blocks is None or not 0 <= layer < len(blocks):
            raise ValueError("Expected model.model.layers with a valid zero-based layer index")
        self.block = blocks[layer]
        self.layer = layer

    def _inputs(self, prompt):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return {key: value.to(self.network.device) for key, value in self.tokenizer(text, return_tensors="pt").items()}

    def hidden(self, prompt):
        captured = []
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured.append(value[0, -1].detach().float().cpu().tolist())
        handle = self.block.register_forward_hook(capture)
        try:
            with self.torch.no_grad():
                self.network(**self._inputs(prompt), use_cache=False)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Target block did not produce a hidden state")
        return captured[-1]

    def generate_steered(self, prompt, seed, delta):
        handle = self.block.register_forward_hook(steering_hook(self.torch, delta)) if delta is not None else None
        try:
            self.torch.manual_seed(seed)
            inputs = self._inputs(prompt)
            with self.torch.no_grad():
                output = self.network.generate(**inputs, do_sample=True, temperature=self.temperature,
                                               top_p=1.0, max_new_tokens=16,
                                               pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            if handle is not None:
                handle.remove()

    def generate(self, prompt, seed):
        return self.generate_steered(prompt, seed, None)
