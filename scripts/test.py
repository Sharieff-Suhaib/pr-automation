# test_lora.py

import os
import time

# ============================================================
# IMPORTANT: MPS FALLBACK
# Must be set BEFORE importing torch
# ============================================================

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "codellama/CodeLlama-7b-hf"

ADAPTER_PATH = (
    "/Users/anieshwarsaravanan/pr-automation/"
    "adapters/repairllama-bugsinpy-lora"
)

# Start small for testing.
# Once this works, increase to 50/100/etc.
MAX_NEW_TOKENS = 20


# ============================================================
# DEVICE DETECTION
# ============================================================

print("\n" + "=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():

    DEVICE = "cuda"

    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))

elif torch.backends.mps.is_available():

    DEVICE = "mps"

    print("CUDA available: NO")
    print("Apple MPS available: YES")
    print("Using Apple GPU (MPS)")

else:

    DEVICE = "cpu"

    print("CUDA available: NO")
    print("Apple MPS available: NO")
    print("Using CPU")

print("Device:", DEVICE)


# ============================================================
# CHECK ADAPTER
# ============================================================

print("\n" + "=" * 60)
print("CHECKING LoRA ADAPTER")
print("=" * 60)

if not os.path.exists(ADAPTER_PATH):

    raise FileNotFoundError(
        f"\nLoRA adapter not found:\n{ADAPTER_PATH}"
    )

print("Adapter path:")
print(ADAPTER_PATH)

print("\nAdapter files:")

for file in os.listdir(ADAPTER_PATH):
    print("   ", file)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(
    f"Tokenizer loaded successfully "
    f"({time.time() - start_time:.2f}s)"
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING CODELLAMA-7B")
print("=" * 60)

print("This can take some time...")

start_time = time.time()

if DEVICE == "mps":

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16
    )

    print("Base model loaded into CPU memory")

    print("Moving model to Apple GPU (MPS)...")

    base_model = base_model.to("mps")

elif DEVICE == "cuda":

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto"
    )

else:

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32
    )

    base_model = base_model.to("cpu")


print(
    f"Base CodeLlama loaded successfully "
    f"({time.time() - start_time:.2f}s)"
)


# ============================================================
# LOAD LoRA ADAPTER
# ============================================================

print("\n" + "=" * 60)
print("LOADING LoRA ADAPTER")
print("=" * 60)

start_time = time.time()

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

print(
    f"LoRA adapter loaded successfully "
    f"({time.time() - start_time:.2f}s)"
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()

print("Model set to evaluation mode")

print(
    "Model device:",
    next(model.parameters()).device
)

print(
    "Model dtype:",
    next(model.parameters()).dtype
)


# ============================================================
# TEST MPS
# ============================================================

if DEVICE == "mps":

    print("\n" + "=" * 60)
    print("TESTING MPS")
    print("=" * 60)

    try:

        mps_test = torch.tensor(
            [[1.0, 2.0, 3.0]],
            dtype=torch.float16,
            device="mps"
        )

        mps_result = mps_test @ mps_test.T

        # Force synchronization
        torch.mps.synchronize()

        print("MPS test successful")
        print("Result:", mps_result)

    except Exception as e:

        print("MPS test failed")
        print(type(e).__name__)
        print(str(e))

        raise


# ============================================================
# TEST PROMPT
# ============================================================

prompt = """Write a Python function to reverse a string.
Return only the Python code."""


print("\n" + "=" * 60)
print("TEST PROMPT")
print("=" * 60)

print(prompt)


# ============================================================
# TOKENIZE
# ============================================================

print("\n" + "=" * 60)
print("TOKENIZING")
print("=" * 60)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

print(
    "Input token count:",
    inputs["input_ids"].shape[1]
)


# ============================================================
# MOVE INPUT TO DEVICE
# ============================================================

inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}

if DEVICE == "mps":
    torch.mps.synchronize()

print("Input moved to:", DEVICE)


# ============================================================
# FORWARD PASS TEST
# ============================================================

print("\n" + "=" * 60)
print("TESTING FORWARD PASS")
print("=" * 60)

print("Running one forward pass...")

start_time = time.time()

try:

    with torch.inference_mode():

        test_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True
        )

    if DEVICE == "mps":
        torch.mps.synchronize()

    forward_time = time.time() - start_time

    print(
        f"Forward pass successful "
        f"({forward_time:.2f}s)"
    )

    print(
        "Logits shape:",
        test_output.logits.shape
    )

    # Free forward-pass output
    del test_output

except Exception as e:

    print("\n" + "=" * 60)
    print("FORWARD PASS ERROR")
    print("=" * 60)

    print(type(e).__name__)
    print(str(e))

    raise


# ============================================================
# GENERATION
# ============================================================

print("\n" + "=" * 60)
print("STARTING GENERATION")
print("=" * 60)

print(
    f"Maximum new tokens: {MAX_NEW_TOKENS}"
)

print("Sampling: disabled")
print("KV cache: enabled")
print("Please wait...")

start_time = time.time()

try:

    with torch.inference_mode():

        outputs = model.generate(

            input_ids=inputs["input_ids"],

            attention_mask=inputs["attention_mask"],

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            # IMPORTANT
            # KV cache makes autoregressive generation
            # much faster than use_cache=False.
            use_cache=True,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    if DEVICE == "mps":
        torch.mps.synchronize()

    generation_time = time.time() - start_time

except Exception as e:

    print("\n" + "=" * 60)
    print("GENERATION ERROR")
    print("=" * 60)

    print("Error type:")
    print(type(e).__name__)

    print("\nError message:")
    print(str(e))

    raise


# ============================================================
# GENERATION FINISHED
# ============================================================

print("\n" + "=" * 60)
print("GENERATION FINISHED")
print("=" * 60)

print(
    f"Generation time: {generation_time:.2f} seconds"
)

print(
    "Total output tokens:",
    outputs.shape[1]
)


# ============================================================
# REMOVE PROMPT FROM OUTPUT
# ============================================================

input_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[0][input_length:]

print(
    "New tokens generated:",
    generated_tokens.shape[0]
)


# ============================================================
# DECODE
# ============================================================

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("LoRA MODEL RESPONSE")
print("=" * 60)

if response.strip():

    print(response)

else:

    print("[EMPTY RESPONSE]")


# ============================================================
# PERFORMANCE
# ============================================================

if generated_tokens.shape[0] > 0:

    tokens_per_second = (
        generated_tokens.shape[0] / generation_time
    )

    print("\n" + "=" * 60)
    print("PERFORMANCE")
    print("=" * 60)

    print(
        f"Generation speed: "
        f"{tokens_per_second:.2f} tokens/sec"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)