# import torch
# from colpali_engine.models import BiGemma3, BiGemmaProcessor3

# MODEL_NAME = "Cognitive-Lab/NetraEmbed"

# def inspect_entire_model():
#     print("Loading NetraEmbed model and processor for deep inspection...")
#     model = BiGemma3.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
#     processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME)
#     model.eval()

#     print("\n" + "="*50)
#     print("1. MODEL CONFIGURATION PARAMETERS")
#     print("="*50)
#     print(model.config)

#     print("\n" + "="*50)
#     print("2. TOP-LEVEL SUB-MODULES (THE ENTIRE ARCHITECTURE)")
#     print("="*50)
#     for name, module in model.named_children():
#         print(f"Module Name: {name} | Type: {type(module).__name__}")

#     print("\n" + "="*50)
#     print("3. DETAILED LAYER BREAKDOWN & PARAMETER COUNTS")
#     print("="*50)
#     total_params = 0
#     trainable_params = 0
#     for name, param in model.named_parameters():
#         num_params = param.numel()
#         total_params += num_params
#         if param.requires_grad:
#             trainable_params += num_params
#         # Print sample layer details
#         if "vision_model" in name or "language_model" in name or "proj" in name:
#             if "weight" in name and ('0.weight' in name or 'embed_tokens' in name):
#                 print(f"Layer: {name} | Shape: {param.shape} | Dtype: {param.dtype}")

#     print(f"\nTotal Parameters: {total_params:,}")
#     print(f"Trainable Parameters: {trainable_params:,}")

#     print("\n" + "="*50)
#     print("4. PROCESSOR & TOKENIZER INSPECTION")
#     print("="*50)
#     print(f"Processor Type: {type(processor)}")
#     if hasattr(processor, "tokenizer"):
#         print(f"Vocab Size: {processor.tokenizer.vocab_size}")
#         print(f"Model Max Length: {processor.tokenizer.model_max_length}")

#     print("\n" + "="*50)
#     print("5. FORWARD PASS TENSOR INSPECTION (DUMMY RUN)")
#     print("="*50)
#     # Test query input trace
#     dummy_query = processor.process_queries(["Test historical query"]).to(model.device)
#     with torch.no_grad():
#         q_out = model(**dummy_query)
#     print(f"Query Forward Output Shape: {q_out.shape}")
#     print(f"Query Output Device: {q_out.device}")
#     print(f"Query Output Dtype: {q_out.dtype}")

#     print("\nModel internal inspection completed successfully!")

# if __name__ == "__main__":
#     inspect_entire_model()

import torch
from PIL import Image
from colpali_engine.models import BiGemma3, BiGemmaProcessor3

MODEL_NAME = "Cognitive-Lab/NetraEmbed"

def deep_tensor_inspection():
    print("Loading model for deep tensor tracing...")
    model = BiGemma3.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = BiGemmaProcessor3.from_pretrained(MODEL_NAME)
    model.eval()

    # Hook dictionary to capture intermediate activations
    activation_records = {}

    def get_hook(name):
        def hook(module, input, output):
            # Capture tensor shapes and types dynamically during forward pass
            if isinstance(output, tuple):
                activation_records[name] = [o.shape if isinstance(o, torch.Tensor) else type(o) for o in output]
            elif isinstance(output, torch.Tensor):
                activation_records[name] = output.shape
            else:
                activation_records[name] = type(output)
        return hook

    # Register hooks on core sub-components of Gemma-3 multimodal architecture
    for name, module in model.named_modules():
        if any(target in name for target in ["vision_model", "language_model", "multi_modal_projector", "embed_tokens"]):
            module.register_forward_hook(get_hook(name))

    # Create dummy multimodal inputs
    dummy_query = processor.process_queries(["Inspect hidden token representation"]).to(model.device)
    
    print("\n--- RUNNING FORWARD PASS & TRACING INTERNAL TENSORS ---")
    with torch.no_grad():
        output = model(**dummy_query)

    print(f"\nFinal Model Output Tensor Shape: {output.shape}")
    print(f"Final Model Output Tensor Dtype: {output.dtype}")
    print(f"Final Model Output Device: {output.device}")

    print("\n--- CAPTURED INTERMEDIATE LAYER SHAPES ---")
    for layer_name, shape_info in list(activation_records.items())[:25]:  # Display first 25 core layers
        print(f"Layer: {layer_name} | Output Shape/Type: {shape_info}")

if __name__ == "__main__":
    deep_tensor_inspection()