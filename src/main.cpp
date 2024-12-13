#include "npc.h"
#include <iostream>
#include <string>

int main() {
    // Create NPC with basic backstory
    NPC npc("Ada", "I am Ada, a merchant in the coastal town of Seavale. "
                   "I run a small shop selling herbs and magical ingredients. "
                   "I've lived here for 15 years and know most of the townspeople.");

    std::cout << "Talking to Ada (type 'quit' to exit)\n";
    std::cout << "----------------------------------------\n";

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

        // Add to NPC's memory
        npc.addMemory("User said: " + input, "conversation");
    }

    return 0;
}
