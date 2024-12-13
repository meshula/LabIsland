#pragma once
#include <string>
#include <vector>
#include <memory>

class NPC {
public:
    struct Memory {
        std::string content;
        long timestamp;
        std::string type;  // conversation, action, need
    };

    struct Need {
        std::string name;
        float value;  // 0.0 to 1.0
    };

    NPC(const std::string& name, const std::string& backstory);
    
    std::string processInput(const std::string& input);
    void addMemory(const std::string& content, const std::string& type);
    void updateNeed(const std::string& need, float value);

private:
    std::string name_;
    std::string backstory_;
    std::vector<Memory> memories_;
    std::vector<Need> needs_;
    // TODO: Add LLM model member once we integrate ONNX/TFLite
};
