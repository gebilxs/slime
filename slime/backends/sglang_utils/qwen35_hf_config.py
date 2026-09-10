"""Keep SGLang on pod transformers 4.57 while Megatron uses the TF 5.15 overlay.

TF 5.15 AutoConfig + sglang's Qwen3.5 MoE config leaves ``text_config`` as a
dict (``get_hf_text_config`` asserts) and drops fields such as
``norm_topk_prob``. SGLang engines therefore strip ``tf515*`` from
PYTHONPATH; the hydrate patch is a fallback for leftover dict configs.
"""

import os


def sglang_pythonpath(pp: str | None = None, hook_dir: str | None = None) -> str:
    parts = [p for p in (pp if pp is not None else os.environ.get("PYTHONPATH", "")).split(":") if p and "tf515" not in p]
    if hook_dir and hook_dir not in parts:
        parts.insert(0, hook_dir)
    return ":".join(parts)


def hydrate_sglang_nested_config(config):
    sub_configs = getattr(type(config), "sub_configs", None)
    if not isinstance(sub_configs, dict):
        return config
    for key, cls in sub_configs.items():
        value = getattr(config, key, None)
        if isinstance(value, dict) and cls is not None:
            try:
                setattr(config, key, cls(**value))
            except TypeError:
                pass
    return config


def patch_sglang_hf_text_config() -> None:
    import sglang.srt.utils.hf_transformers_utils as hf_utils

    orig = hf_utils.get_hf_text_config
    if getattr(orig, "_slime_hydrate_text_config", False):
        return

    def wrapped(config):
        hydrate_sglang_nested_config(config)
        return orig(config)

    wrapped._slime_hydrate_text_config = True
    hf_utils.get_hf_text_config = wrapped
    for mod_name in ("sglang.srt.configs.model_config", "sglang.srt.server_args"):
        try:
            mod = __import__(mod_name, fromlist=["get_hf_text_config"])
        except Exception:
            continue
        if getattr(mod, "get_hf_text_config", None) is orig:
            setattr(mod, "get_hf_text_config", wrapped)
