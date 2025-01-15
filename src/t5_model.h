#pragma once
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

#include "tokenizer.h"

class T5Model {
public:
    T5Model(const std::string& model_path, const std::string& sp_model_path);
    ~T5Model() = default;
    std::string infer(const std::string& input_text);

protected:
    // Convert tokens to input tensors for the model
    std::vector<Ort::Value> tokens_to_tensors(const std::vector<int64_t>& tokens);
    
    // Convert model output tensor to next token
    int64_t tensor_to_next_token(const Ort::Value& output_tensor, size_t current_len);

protected:
    Tokenizer tokenizer_;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

    int pad_token_id_;
    int eos_token_id_;
    int unk_token_id_;
};
