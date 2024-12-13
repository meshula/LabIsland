#pragma once
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

class T5Model {
public:
    T5Model(const std::string& model_path);
    ~T5Model();

    std::string generate(const std::string& input_text, size_t max_length = 128);

private:
    std::vector<int64_t> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int64_t>& tokens);
    
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    
    // Memory allocations for input/output tensors
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    
    // Model configuration
    const size_t max_sequence_length_ = 128;
    const size_t batch_size_ = 1;
    
    // Prevent copying
    T5Model(const T5Model&) = delete;
    T5Model& operator=(const T5Model&) = delete;
};
