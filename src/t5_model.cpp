#include "t5_model.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

static constexpr size_t max_sequence_length_ = 64;  // Reduced to avoid memory issues
static constexpr size_t batch_size_ = 1;
static constexpr size_t min_tokens_ = 1;

T5Model::T5Model(const std::string& model_path, const std::string& sp_model_path)
: tokenizer_(sp_model_path) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "T5Model");
    allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // Initialize special token IDs from the tokenizer
    pad_token_id_ = tokenizer_.GetPadTokenId();
    eos_token_id_ = tokenizer_.GetEosTokenId();
    unk_token_id_ = tokenizer_.GetUnkTokenId();
    
    std::cerr << "Initialized with special tokens:\n"
              << "  PAD=" << pad_token_id_ << "\n"
              << "  EOS=" << eos_token_id_ << "\n"
              << "  UNK=" << unk_token_id_ << "\n";
}


std::string T5Model::infer(const std::string& input_text) {
    // T5 expects a task prefix
    std::string prefixed_input = "translate English to English: " + input_text;
    std::vector<int64_t> tokens = tokenizer_.tokenize(prefixed_input);
    
    // Debug input tokens
    std::cerr << "Input text: '" << prefixed_input << "'\n";
    std::cerr << "Input tokens: ";
    for (auto token : tokens) {
        std::cerr << token << " ";
    }
    std::cerr << "\n";
    if (tokens.size() < min_tokens_)
        return "";
    // Add EOS token to input sequence
    tokens.push_back(eos_token_id_);
    if (tokens.size() >= max_sequence_length_) {
        tokens.resize(max_sequence_length_ - 1);
        tokens.push_back(eos_token_id_);  // Ensure EOS is the last token
    }

    // Get vocabulary size from the tokenizer
    size_t vocab_size = tokenizer_.GetVocabSize();
    std::cerr << "Vocabulary size: " << vocab_size << "\n";

    std::vector<int64_t> input_ids(max_sequence_length_, pad_token_id_);
    std::vector<int64_t> attention_mask(max_sequence_length_, 0);
    // Initialize decoder input with start token (pad token for T5)
    std::vector<int64_t> decoder_input_ids(max_sequence_length_, pad_token_id_);
    std::vector<int64_t> decoder_attention_mask(max_sequence_length_, 0);
    decoder_input_ids[0] = pad_token_id_;  // Start with pad token
    decoder_attention_mask[0] = 1;  // Only attend to the first position

    std::cerr << "Using pad token " << pad_token_id_ << " as start token for decoder input\n";

    // Copy tokens and set attention mask
    std::copy(tokens.begin(), tokens.end(), input_ids.begin());
    for (size_t i = 0; i < tokens.size(); i++) {
        attention_mask[i] = 1;
    }

    // Debug input tensors
    std::cerr << "Input IDs: ";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cerr << input_ids[i] << " ";
    }
    std::cerr << "\nAttention mask: ";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cerr << attention_mask[i] << " ";
    }
    std::cerr << "\nDecoder input IDs: ";
    for (size_t i = 0; i < 5; i++) {  // Just show first few tokens
        std::cerr << decoder_input_ids[i] << " ";
    }
    std::cerr << "\nDecoder attention mask: ";
    for (size_t i = 0; i < 5; i++) {  // Just show first few tokens
        std::cerr << decoder_attention_mask[i] << " ";
    }
    std::cerr << "\n";

    // Create input tensor shapes
    std::vector<int64_t> shape = {1, static_cast<int64_t>(max_sequence_length_)};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), shape.data(), shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(), shape.data(), shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, decoder_input_ids.data(), decoder_input_ids.size(), shape.data(), shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, decoder_attention_mask.data(), decoder_attention_mask.size(), shape.data(), shape.size()));

    const char* input_node_names[] = {"input_ids", "attention_mask",
        "decoder_input_ids", "decoder_attention_mask", nullptr};
    const char* output_node_names[] = {"output", nullptr};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names,
        input_tensors.data(),
        input_tensors.size(),
        output_node_names,
        1); //output_node_names_.size());

    auto& output_tensor = output_tensors.front();
    float* logits = output_tensor.GetTensorMutableData<float>();
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    // Expected shape: [batch_size, sequence_length, vocab_size]
    if (output_shape.size() != 3) {
        throw std::runtime_error("Unexpected output tensor shape");
    }
    
    size_t batch_size = output_shape[0];
    size_t seq_len = output_shape[1];
    size_t model_vocab_size = output_shape[2];
    
    std::cerr << "Model output shape: [" << batch_size << ", " << seq_len << ", " << model_vocab_size << "]\n";
    if (model_vocab_size != vocab_size) {
        std::cerr << "Warning: Model vocab size (" << model_vocab_size 
                  << ") differs from tokenizer vocab size (" << vocab_size << ")\n";
    }

    // Process output logits one token at a time
    std::vector<int64_t> output_tokens;
    for (size_t i = 0; i < seq_len && output_tokens.size() < max_sequence_length_; i++) {
        std::cerr << "Processing position " << i << " with " << output_tokens.size() << " tokens generated\n";
        // Calculate offset for current position in batch 0
        // For a tensor of shape [batch_size, seq_len, vocab_size], 
        // to access element [b, s, v] we need offset = (s * batch_size * vocab_size) + (b * vocab_size) + v
        size_t base_offset = (i * batch_size * model_vocab_size) + (0 * model_vocab_size);
        
        // Apply softmax and find max probability token
        float max_prob = 0.0f;
        size_t max_idx = 0;
        float sum_exp = 0.0f;
        std::vector<float> probs(model_vocab_size);
        
        // First pass: compute exp and sum
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < model_vocab_size; j++) {
            float logit = logits[base_offset + j];
            if (logit > max_logit) max_logit = logit;
        }
        
        for (size_t j = 0; j < model_vocab_size; j++) {
            float logit = logits[base_offset + j] - max_logit;  // Subtract max for numerical stability
            probs[j] = std::exp(logit);
            sum_exp += probs[j];
        }
        
        // Second pass: normalize and find max probability
        for (size_t j = 0; j < model_vocab_size; j++) {
            probs[j] /= sum_exp;
            if (probs[j] > max_prob) {
                max_prob = probs[j];
                max_idx = j;
            }
        }

        // Debug output
        std::cerr << "Position " << i << ": token=" << max_idx 
                  << " prob=" << max_prob 
                  << " (eos=" << eos_token_id_ 
                  << " pad=" << pad_token_id_ 
                  << " base_offset=" << base_offset
                  << " shape=" << output_shape[0] << "," << output_shape[1] << "," << output_shape[2]
                  << ")\n";
        
        // Stop at EOS token
        if (max_idx == static_cast<size_t>(eos_token_id_)) {
            std::cerr << "Found EOS token\n";
            break;
        }
        
        output_tokens.push_back(static_cast<int64_t>(max_idx));
    }
    return tokenizer_.detokenize(output_tokens);
}
