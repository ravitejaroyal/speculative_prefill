from speculative_prefill.hf_speculative_prefill import HFFSpeculativePrefill

if __name__ == "__main__":
    sp = HFFSpeculativePrefill(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        spec_model="meta-llama/Llama-3.2-1B-Instruct",
        look_ahead=4,
        keep_percentage=0.5,
        pool_kernel_size=1,
        device="cuda"
    )
    prompt = "Tell me about the city of Chicago."
    logits = sp(prompt)
    print(logits.shape)
