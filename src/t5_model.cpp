#include "t5_model.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

bool T5Model::isValidTokenId(int64_t id) const {
    auto vocab_size = tokenizer_->GetPieceSize();
    if (debug_mode_ && (id < 0 || id >= static_cast<int64_t>(vocab_size))) {
        std::cout << "Invalid token ID " << id << " (valid range: 0 to " << (vocab_size - 1) << ")" << std::endl;
    }
    return id >= 0 && id < static_cast<int64_t>(vocab_size);
}

T5Model::T5Model(const std::string& model_path) {
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "T5Model");
    allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

    // Session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Create session
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // Set up input and output names
    input_node_names_ = {"input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"};
    output_node_names_ = {"output"};

    // Initialize SentencePiece tokenizer
    tokenizer_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    std::filesystem::path model_dir = std::filesystem::path(model_path).parent_path();
    std::string spiece_model_path = (model_dir / "spiece.model").string();
    
    if (!tokenizer_->Load(spiece_model_path).ok()) {
        throw std::runtime_error("Failed to load SentencePiece model from: " + spiece_model_path);
    }

    // Cache special token IDs
    pad_token_id_ = tokenizer_->PieceToId("<pad>");
    eos_token_id_ = tokenizer_->PieceToId("</s>");
    unk_token_id_ = tokenizer_->PieceToId("<unk>");

    if (!isValidTokenId(pad_token_id_) || !isValidTokenId(eos_token_id_) || !isValidTokenId(unk_token_id_)) {
        throw std::runtime_error("Invalid special token IDs");
    }

    if (debug_mode_) {
        std::cout << "Model initialized. Special tokens - PAD: " << pad_token_id_ 
                  << ", EOS: " << eos_token_id_ 
                  << ", UNK: " << unk_token_id_ << std::endl;
        std::cout << "Vocabulary size: " << tokenizer_->GetPieceSize() << std::endl;
    }
}

T5Model::~T5Model() = default;

std::string T5Model::generate(const std::string& input_text, size_t max_length) {
    try {
        if (debug_mode_) {
            std::cout << "Starting generation..." << std::endl;
        }
        
        // Tokenize input text
        auto tokens = tokenize(input_text);
        
        // Ensure we have at least one token
        if (tokens.size() < min_tokens_) {
            if (debug_mode_) {
                std::cout << "Input text too short after tokenization" << std::endl;
            }
            return "";
        }
        
        // Truncate if needed
        if (tokens.size() >= max_sequence_length_) {
            if (debug_mode_) {
                std::cout << "Truncating input tokens from " << tokens.size() 
                         << " to " << max_sequence_length_ - 1 << std::endl;
            }
            tokens.resize(max_sequence_length_ - 1);
            tokens.push_back(eos_token_id_);
        }

        // Create input tensors
        std::vector<int64_t> input_ids(max_sequence_length_, pad_token_id_);
        std::vector<int64_t> attention_mask(max_sequence_length_, 0);
        std::vector<int64_t> decoder_input_ids(max_sequence_length_, pad_token_id_);
        std::vector<int64_t> decoder_attention_mask(max_sequence_length_, 0);

        // Copy tokens and set attention mask
        std::copy(tokens.begin(), tokens.end(), input_ids.begin());
        for (size_t i = 0; i < tokens.size(); i++) {
            attention_mask[i] = 1;
        }
        
        // Set initial decoder token and mask
        decoder_input_ids[0] = pad_token_id_;
        decoder_attention_mask[0] = 1;

        if (debug_mode_) {
            std::cout << "Preparing tensors..." << std::endl;
            std::cout << "Sequence length: " << max_sequence_length_ << std::endl;
        }

        // Create input tensor shapes
        std::vector<int64_t> shape = {1, static_cast<int64_t>(max_sequence_length_)};

        if (debug_mode_) {
            std::cout << "Tensor shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
        }

        // Create input tensors
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

        if (debug_mode_) {
            std::cout << "Running inference..." << std::endl;
        }

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_node_names_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_node_names_.data(),
            output_node_names_.size());

        if (debug_mode_) {
            std::cout << "Processing output..." << std::endl;
        }

        // Get output tensor info
        auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto output_shape = tensor_info.GetShape();
        
        if (debug_mode_) {
            std::cout << "Output tensor shape: [";
            for (size_t i = 0; i < output_shape.size(); i++) {
                std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        // Get output data as int32_t instead of int64_t
        const int32_t* output_data = output_tensors[0].GetTensorData<int32_t>();
        size_t output_size = tensor_info.GetElementCount();
        
        if (debug_mode_) {
            std::cout << "Output tensor size: " << output_size << std::endl;
        }

        // Safely copy and validate output data
        std::vector<int64_t> output_tokens;
        output_tokens.reserve(output_size);
        
        for (size_t i = 0; i < output_size; i++) {
            // Safely convert int32_t to int64_t
            int64_t token_id = static_cast<int64_t>(output_data[i]);
            if (isValidTokenId(token_id)) {
                output_tokens.push_back(token_id);
            } else if (debug_mode_) {
                std::cout << "Invalid token ID " << token_id << " at position " << i << std::endl;
            }
        }

        if (output_tokens.empty()) {
            if (debug_mode_) {
                std::cout << "No valid output tokens generated" << std::endl;
            }
            return "";
        }

        if (debug_mode_) {
            std::cout << "Valid output tokens (" << output_tokens.size() << "): ";
            for (size_t i = 0; i < std::min(output_tokens.size(), size_t(10)); i++) {
                std::cout << output_tokens[i] << " ";
            }
            if (output_tokens.size() > 10) std::cout << "...";
            std::cout << std::endl;
        }

        // Filter out padding and handle special tokens
        std::vector<int64_t> filtered_tokens;
        filtered_tokens.reserve(output_tokens.size());
        
        for (const auto& token : output_tokens) {
            if (token == pad_token_id_) continue;
            if (token == eos_token_id_) break;
            filtered_tokens.push_back(token);
        }

        if (filtered_tokens.empty()) {
            if (debug_mode_) {
                std::cout << "No tokens remaining after filtering" << std::endl;
            }
            return "";
        }

        if (debug_mode_) {
            std::cout << "Filtered tokens (" << filtered_tokens.size() << "): ";
            for (size_t i = 0; i < std::min(filtered_tokens.size(), size_t(10)); i++) {
                std::cout << filtered_tokens[i] << " ";
            }
            if (filtered_tokens.size() > 10) std::cout << "...";
            std::cout << std::endl;
        }

        // Detokenize output
        return detokenize(filtered_tokens);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return "Error generating response";
    }
    catch (const std::exception& e) {
        std::cerr << "Error during generation: " << e.what() << std::endl;
        return "Error generating response";
    }
}

std::vector<int64_t> T5Model::tokenize(const std::string& text) {
    if (debug_mode_) {
        std::cout << "Tokenizing input text..." << std::endl;
    }
    
    std::vector<int> piece_ids;
    if (!tokenizer_->Encode(text, &piece_ids).ok()) {
        throw std::runtime_error("Failed to tokenize text: " + text);
    }
    
    // Convert int to int64_t and validate
    std::vector<int64_t> tokens;
    tokens.reserve(piece_ids.size());
    for (int id : piece_ids) {
        int64_t token_id = static_cast<int64_t>(id);
        if (isValidTokenId(token_id)) {
            tokens.push_back(token_id);
        } else if (debug_mode_) {
            std::cout << "Skipping invalid token ID in tokenize: " << token_id << std::endl;
        }
    }

    if (debug_mode_) {
        std::cout << "Input tokens (" << tokens.size() << "): ";
        for (size_t i = 0; i < std::min(tokens.size(), size_t(10)); i++) {
            std::cout << tokens[i] << " ";
        }
        if (tokens.size() > 10) std::cout << "...";
        std::cout << std::endl;
    }
    
    return tokens;
}

std::string T5Model::detokenize(const std::vector<int64_t>& tokens) {
    if (debug_mode_) {
        std::cout << "Detokenizing output..." << std::endl;
    }
    
    if (tokens.empty()) {
        return "";
    }

    // Convert int64_t to int and validate
    std::vector<int> piece_ids;
    piece_ids.reserve(tokens.size());
    
    for (int64_t id : tokens) {
        if (isValidTokenId(id)) {
            piece_ids.push_back(static_cast<int>(id));
        } else if (debug_mode_) {
            std::cout << "Skipping invalid token ID in detokenize: " << id << std::endl;
        }
    }
    
    if (piece_ids.empty()) {
        return "";
    }
    
    std::string text;
    if (!tokenizer_->Decode(piece_ids, &text).ok()) {
        if (debug_mode_) {
            std::cerr << "Failed to decode tokens: ";
            for (size_t i = 0; i < std::min(piece_ids.size(), size_t(10)); i++) {
                std::cerr << piece_ids[i] << " ";
            }
            if (piece_ids.size() > 10) std::cerr << "...";
            std::cerr << std::endl;
        }
        throw std::runtime_error("Failed to detokenize tokens");
    }
    
    return text;
}
