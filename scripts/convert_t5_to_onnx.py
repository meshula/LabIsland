import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

def convert_t5_to_onnx(model_name="google/t5-efficient-tiny", output_path="models/t5_tiny.onnx"):
    print(f"Loading {model_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    encoder_seq_length = 128
    decoder_seq_length = 128
    
    # Encoder inputs
    input_ids = torch.ones(batch_size, encoder_seq_length, dtype=torch.long)
    attention_mask = torch.ones(batch_size, encoder_seq_length, dtype=torch.long)
    
    # Decoder inputs
    decoder_input_ids = torch.ones(batch_size, decoder_seq_length, dtype=torch.long)
    decoder_attention_mask = torch.ones(batch_size, decoder_seq_length, dtype=torch.long)
    
    # Export the model
    print("Converting to ONNX...")
    
    # Define dynamic axes
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'encoder_sequence'},
        'attention_mask': {0: 'batch_size', 1: 'encoder_sequence'},
        'decoder_input_ids': {0: 'batch_size', 1: 'decoder_sequence'},
        'decoder_attention_mask': {0: 'batch_size', 1: 'decoder_sequence'},
        'output': {0: 'batch_size', 1: 'decoder_sequence'}
    }
    
    # Export with all necessary inputs and higher opset version
    torch.onnx.export(
        model,
        (
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask
        ),
        output_path,
        input_names=[
            'input_ids',
            'attention_mask',
            'decoder_input_ids',
            'decoder_attention_mask'
        ],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=14,  # Increased from 12 to 14
        do_constant_folding=True
    )
    print(f"Model converted and saved to {output_path}")
    
    # Save tokenizer config
    tokenizer_config = {
        "vocab_file": tokenizer.vocab_file if hasattr(tokenizer, 'vocab_file') else None,
        "model_max_length": tokenizer.model_max_length,
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "unk_token": tokenizer.unk_token
    }
    
    # Save tokenizer configuration
    import json
    config_path = os.path.join(os.path.dirname(output_path), "tokenizer_config.json")
    with open(config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Tokenizer config saved to {config_path}")

if __name__ == "__main__":
    convert_t5_to_onnx()
