
This project is about creating a game NPC. The game NPC will have a memory of conversations, and a memory of actions it's taken, such as going to a location, acquiring items, having conversations with other player and non player characters, and a sense of self monitoring essential needs.

It will use a small or tiny T5 model running in C or C++ using ONNX Runtime or TensorFlow Lite C API.

Initial steps are:

1. Convert the T5 model to ONNX or TFLite.
2. Use ONNX Runtime or TensorFlow Lite C API for inference in C++.
3. Create an NPC object to contain an instance of the LLM.
4. Give the NPC a small backstory
5. Create a small REPL where a human user can interact with the NPC by asking it questions, 

