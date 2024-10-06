"""
    Patch 1: Add monkey patch to llama that allows using the offset position ids
        e.g.
        Original position ids: [0, 1, 2, 3, ..., 15]
        Kept position ids: [0, 1, 4, 8, 13]
        Prefill position ids: [0, 1, 4, 8, 13]
        Decode position ids: [0, 1, 4, 8, 13] + [16, 17, ...]

    Patch 2: Being able to update the full KV cache
        e.g.
        Speculative phase we have KV for [0, 1, 4, 8, 13] + [16, 17, ...]
        After updating we have KV for [0, 1, 2, 3, ..., 15] + [16, 17, ...]
"""

