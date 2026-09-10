"""Auto-applied on every SGLang process that has this directory on PYTHONPATH."""

try:
    from slime.backends.sglang_utils.qwen35_hf_config import patch_sglang_hf_text_config

    patch_sglang_hf_text_config()
except Exception:
    pass
