#include "npc.h"
#include <chrono>

NPC::NPC(const std::string& name, const std::string& backstory)
    : name_(name), backstory_(backstory) {
    // Initialize basic needs
    needs_ = {
        {"hunger", 1.0f},
        {"energy", 1.0f},
        {"social", 1.0f}
    };
}

std::string NPC::processInput(const std::string& input) {
    // TODO: Replace with actual LLM inference
    return "I hear you, but I'm still learning how to respond properly.";
}

void NPC::addMemory(const std::string& content, const std::string& type) {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    
    memories_.push_back({content, timestamp, type});
    
    // Keep only recent memories (last 100 for now)
    if (memories_.size() > 100) {
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
