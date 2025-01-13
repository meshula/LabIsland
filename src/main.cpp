#include <iostream>
#include <string>
#include <filesystem>

#include "t5_model.h"

int main(int argc, char* argv[]) {
    // Get the path to the ONNX model relative to the executable
    std::filesystem::path model_path = std::filesystem::current_path() / "models" / "t5_tiny.onnx";
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Error: Could not find model file at " << model_path << std::endl;
        return 1;
    }

    std::filesystem::path sppath = std::filesystem::current_path() / "models" / "spiece.model";
    if (!std::filesystem::exists(sppath)) {
        std::cerr << "Error: Could not find SentencePiece model file at " << sppath << std::endl;
        return 1;
    }

    T5Model model(model_path.string(), sppath.string());

    std::string user_input;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, user_input);
        if (user_input == "exit") break;

        std::string response = model.infer(user_input);
        std::cout << "Bot: " << response << std::endl;
    }
    return 0;
}
#if 0
    // Create NPC with basic backstory and model
    NPC npc("Ada", 
            "I am Ada, a merchant in the coastal town of Seavale. "
            "I run a small shop selling herbs and magical ingredients. "
            "I've lived here for 15 years and know most of the townspeople.",
            model_path.string());
#endif
