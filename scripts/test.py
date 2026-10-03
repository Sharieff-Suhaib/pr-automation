import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "codellama/CodeLlama-7b-hf"

# Change this to your actual LoRA adapter folder
ADAPTER_PATH = "../adapters/repairllama-bugsinpy-lora"


# ============================================================
# CHECK DEVICE
# ============================================================

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("CUDA available: NO")
    print("Using CPU")

print()


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded")
print()


# ============================================================
# LOAD BASE CODELLAMA
# ============================================================

print("=" * 60)
print("LOADING CODELLAMA-7B")
print("=" * 60)

if device == "cuda":

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

else:

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

print("Base CodeLlama loaded")
print()


# ============================================================
# LOAD LoRA ADAPTER
# ============================================================

print("=" * 60)
print("LOADING LoRA ADAPTER")
print("=" * 60)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("LoRA adapter loaded successfully!")
print()


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_response(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # Move inputs to same device as model
    if device == "cuda":
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=300,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    # Remove original prompt from output
    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response


# ============================================================
# TEST PROMPTS
# ============================================================

test_prompts = [

    """Write a Python function to reverse a string.
Return only the Python code.""",

    """Write a Python function to check whether a number is prime.
Return only the Python code.""",

    """Write a C++ function to find the maximum element
in an array.
Return only the C++ code.""",

    """Complete this Python function:

def factorial(n):
""",

    """Fix the bug in this Python code:

def add(a, b):
    return a - b

Return the corrected code only."""
]


# ============================================================
# RUN TESTS
# ============================================================

print("=" * 60)
print("TESTING LoRA MODEL")
print("=" * 60)

for i, prompt in enumerate(test_prompts, start=1):

    print()
    print("=" * 60)
    print(f"TEST {i}")
    print("=" * 60)

    print("\nPROMPT:")
    print(prompt)

    response = generate_response(prompt)

    print("\nMODEL RESPONSE:")
    print(response)

print()
print("=" * 60)
print("TESTING COMPLETE")
print("=" * 60)