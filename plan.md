# Island NPC Project Plan

This project creates a game NPC with memory of conversations, actions (location visits, item acquisitions, conversations), and self-monitoring of essential needs using a T5 model for natural language interaction.

## Phase 1: Model Setup ✓
1. Convert T5-tiny model to ONNX format ✓
   - Script: scripts/convert_t5_to_onnx.py
   - Output: models/t5_tiny.onnx
2. Setup ONNX Runtime integration in C++ ✓
   - Implementation: t5_model.h/cpp
   - Basic model loading and inference structure

## Phase 2: Text Processing (Current Phase)
1. Implement tokenization/detokenization
   - [x] Copy SentencePiece model file to project
   - [x] Implement SentencePiece integration in T5Model
   - [x] Complete tokenize() method in t5_model.cpp
   - [x] Complete detokenize() method in t5_model.cpp
   - [x] Add debug mode to control token inspection output
   - [x] Fix token handling issues:
     - [x] Added token validation with isValidTokenId method
     - [x] Reduced sequence lengths to prevent memory issues
     - [x] Added proper bounds checking against vocabulary size
     - [x] Improved error handling for decoder failures
   - [ ] Test text generation with fixed token handling:
     - [ ] Test with simple inputs
     - [ ] Test with longer context
     - [ ] Verify token sequences are valid
     - [ ] Check model output quality
2. Test text generation
   - [ ] Create basic test cases:
     - [ ] Simple greetings
     - [ ] Basic conversations
     - [ ] Context-aware responses
   - [ ] Verify model outputs:
     - [ ] Check response coherence
     - [ ] Validate context handling
     - [ ] Test memory integration

## Phase 3: NPC Core Implementation
1. Memory System
   - [ ] Design memory structure for:
     - Conversations
     - Actions/Events
     - Location history
     - Item inventory
     - Essential needs status
2. NPC Personality
   - [ ] Create backstory
   - [ ] Define personality traits
   - [ ] Implement response generation with personality context

## Phase 4: Interaction System
1. REPL Interface
   - [ ] Basic command parsing
   - [ ] Response generation
   - [ ] Memory integration
2. Action System
   - [ ] Define possible actions
   - [ ] Implement action processing
   - [ ] Update memory based on actions

## Phase 5: Testing and Refinement
1. System Testing
   - [ ] Test conversation memory
   - [ ] Test action processing
   - [ ] Test personality consistency
2. Performance Optimization
   - [ ] Optimize model inference
   - [ ] Memory management
   - [ ] Response generation speed

## Checkpoint Information
Current Checkpoint: Phase 2.1
- Completed: 
  - Basic model loading and inference structure
  - Initial tokenization implementation
  - Added special token handling
  - Added debug mode control
  - Fixed token validation issues
  - Improved memory management
  - Added robust error handling
- Next Steps: 
  1. Test text generation with fixed token handling
  2. Verify model outputs are coherent
  3. Test with various input lengths
  4. Move to comprehensive text generation testing

Dependencies:
- ONNX Runtime
- SentencePiece
- T5-tiny model

## Debug Instructions
To test token handling:
1. Build the project
2. Run with debug flag: `./island_npc --debug`
3. Enter test inputs to see:
   - Token validation
   - Sequence lengths
   - Tensor shapes
   - Model outputs
4. Check for:
   - Valid token IDs
   - Proper sequence truncation
   - Successful detokenization
   - Coherent responses
