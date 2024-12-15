#include "npc.h"
#include <chrono>
#include <sstream>

NPC::NPC(const std::string& name, const std::string& backstory, const std::string& model_path)
    : name_(name), backstory_(backstory), model_(std::make_unique<T5Model>(model_path)) {
    // Initialize basic needs
    needs_ = {
        {"hunger", 1.0f},
        {"energy", 1.0f},
        {"social", 1.0f}
    };
}

std::string NPC::processInput(const std::string& input) {
    // Format the prompt with context
    std::string prompt = formatPrompt(input);
    
    // Get response from the model
    std::string response = model_->generate(prompt);
    
    // Add to memory
    addMemory("User: " + input + " | Me: " + response, "conversation");
    
    return response;
}

void NPC::addMemory(const std::string& content, const std::string& type) {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    
    memories_.push_back({content, timestamp, type});
    
    // Keep only recent memories (last 5 for now to manage context size)
    if (memories_.size() > 5) {
        memories_.erase(memories_.begin());
    }
}

void NPC::updateNeed(const std::string& need, float value) {
    for (auto& n : needs_) {
        if (n.name == need) {
            n.value = std::max(0.0f, std::min(1.0f, value));
            break;
        }
    }
}

std::string NPC::formatPrompt(const std::string& input) {
    std::stringstream prompt;
    
    // Add shortened backstory
    prompt << "I am " << name_ << ". " << backstory_ << "\n";
    
    // Add simplified needs state
    prompt << "State:";
    for (const auto& need : needs_) {
        prompt << " " << need.name << "=" << need.value;
    }
    prompt << "\n";
    
    // Add recent memories (last 2 only)
    if (!memories_.empty()) {
        prompt << "Recent: ";
        size_t start = memories_.size() > 2 ? memories_.size() - 2 : 0;
        for (size_t i = start; i < memories_.size(); ++i) {
            prompt << memories_[i].content << " ";
        }
        prompt << "\n";
    }
    
    // Add the current input
    prompt << "User: " << input << "\nResponse:";
    
    return prompt.str();
}
