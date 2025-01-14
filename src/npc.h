#pragma once
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "t5_model.h"

struct Memory {
    std::string content;
    std::time_t timestamp;
    std::string type;
};

struct Need {
    std::string name;
    float value;
};

class NPC {
public:
    NPC(const std::string& name, const std::string& backstory, const std::string& model_path);
    
    std::string processInput(const std::string& input);
    void addMemory(const std::string& content, const std::string& type);
    void updateNeed(const std::string& need, float value);
    
    // Debug control
    void setDebugMode(bool enabled) { model_->setDebugMode(enabled); }
    bool getDebugMode() const { return model_->getDebugMode(); }

private:
    std::string formatPrompt(const std::string& input);
    
    std::string name_;
    std::string backstory_;
    std::vector<Memory> memories_;
    std::vector<Need> needs_;
    std::unique_ptr<T5Model> model_;
};
