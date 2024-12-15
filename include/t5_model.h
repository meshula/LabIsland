#pragma once
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

class T5Model {
public:
    T5Model(const std::string& model_path);
    ~T5Model();

    std::string generate(const std::string& input_text, size_t max_length = 128);
    
    // Debug control
    void setDebugMode(bool enabled) { debug_mode_ = enabled; }
    bool getDebugMode() const { return debug_mode_; }

private:
    std::vector<int64_t> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int64_t>& tokens);
    bool isValidTokenId(int64_t id) const;
    
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
    
    // Memory allocations for input/output tensors
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    
    // Model configuration
    static constexpr size_t max_sequence_length_ = 64;  // Reduced to avoid memory issues
    static constexpr size_t batch_size_ = 1;
    static constexpr size_t min_tokens_ = 1;
    
    // Special token IDs
    int pad_token_id_;
    int eos_token_id_;
    int unk_token_id_;
    
    // Debug flag
    bool debug_mode_ = false;
    
    // Prevent copying
    T5Model(const T5Model&) = delete;
    T5Model& operator=(const T5Model&) = delete;
};
