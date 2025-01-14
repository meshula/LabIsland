# Island NPC Project Plan

This project creates a game NPC with memory of conversations, actions (location visits, item acquisitions, conversations), and self-monitoring of essential needs using a T5 model for natural language interaction.

Meta prompt:

Address the TODO's in order, one at a time.
Addressing a TODO means to implement it, write a corresponding unit test, run the unit test and debug and edit until the test passes. When the test passes, prompt to either git commit work since the last commit, or to refine the test further. Naturally a TODO that is to add a test does not need a test for the tst. When the test passes, mark the TODO in this file as done by filling in an X. If further detail in this plan would help with the work, please propose it.

TODOs:

[X] Add a test for the tokenizer verifying that "brillig", "slithy", and "tove" are not recognized tokens.
[ ] Add a tokenizer test that verifies the first paragraph of Jabberwocky tokenizes in accordance with the expected results of the "brillig", "slithy", and "tove" test.
[ ] Factor t5_model to have a protected tokens to tensor function, and tensor to token function.
[ ] Add a token to tensor test that verifies we can tokenize and tensorize the first paragraph of 1984, and then detensorize to tokens and confirm that the paragraph has been reconstructed.

Dependencies:
- ONNX Runtime
- SentencePiece
- T5-tiny model

## Debug Instructions
To run tests:
1. Build the project using, eg, `make`.
2. Run with debug flag: `./[testname] -- model path/to/models/spiece.model
