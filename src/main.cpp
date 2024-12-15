#include "npc.h"
#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char* argv[]) {
    // Check for debug flag
    bool debug_mode = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--debug" || arg == "-d") {
            debug_mode = true;
            break;
        }
    }

    // Get the path to the ONNX model relative to the executable
    std::filesystem::path model_path = std::filesystem::current_path() / "models" / "t5_tiny.onnx";
    
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Error: Could not find model file at " << model_path << std::endl;
        return 1;
    }

    // Create NPC with basic backstory and model
    NPC npc("Ada", 
            "I am Ada, a merchant in the coastal town of Seavale. "
            "I run a small shop selling herbs and magical ingredients. "
            "I've lived here for 15 years and know most of the townspeople.",
            model_path.string());

    // Set debug mode based on command line flag
    npc.setDebugMode(debug_mode);

    std::cout << "Talking to Ada (type 'quit' to exit)\n";
    std::cout << "----------------------------------------\n";
    if (debug_mode) {
        std::cout << "Debug mode enabled\n";
        std::cout << "----------------------------------------\n";
    }

    std::string input;
    while (true) {
        std::cout << "\nYou: ";
        std::getline(std::cin, input);

        if (input == "quit") {
            break;
        }

        // Process input and get response
        std::string response = npc.processInput(input);
        std::cout << "Ada: " << response << std::endl;
    }

    return 0;
}
