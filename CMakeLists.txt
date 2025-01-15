cmake_minimum_required(VERSION 3.12)
project(island_npc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Download and build µnit
FetchContent_Declare(
    munit
    GIT_REPOSITORY https://github.com/nemequ/munit.git
    GIT_TAG master
)
FetchContent_GetProperties(munit)
if(NOT munit_POPULATED)
    FetchContent_Populate(munit)
    
    # Create munit library
    add_library(munit STATIC
        ${munit_SOURCE_DIR}/munit.c
    )
    target_include_directories(munit PUBLIC ${munit_SOURCE_DIR})
endif()

# Download and build SentencePiece
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.1.99
)
# Disable SentencePiece tests and unnecessary components
set(SPM_ENABLE_SHARED OFF CACHE BOOL "")
set(SPM_ENABLE_TENSORFLOW_SHARED OFF CACHE BOOL "")
set(SPM_BUILD_TEST OFF CACHE BOOL "")
FetchContent_MakeAvailable(sentencepiece)

# Download and extract ONNX Runtime
set(ONNXRUNTIME_VERSION "1.20.1")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
set(ONNXRUNTIME_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime")

if(NOT EXISTS "${ONNXRUNTIME_DOWNLOAD_DIR}")
    file(DOWNLOAD "${ONNXRUNTIME_URL}" "${CMAKE_BINARY_DIR}/onnxruntime.tgz"
        SHOW_PROGRESS
        STATUS DOWNLOAD_STATUS
    )
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download ONNX Runtime")
    endif()

    file(MAKE_DIRECTORY "${ONNXRUNTIME_DOWNLOAD_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/onnxruntime.tgz"
        WORKING_DIRECTORY "${ONNXRUNTIME_DOWNLOAD_DIR}"
    )
endif()

# Find the extracted ONNX Runtime directory
file(GLOB ONNXRUNTIME_ROOT "${ONNXRUNTIME_DOWNLOAD_DIR}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")

# Add executable
add_executable(island_npc 
    src/main.cpp
    src/tokenizer.h
    src/tokenizer.cpp
    src/t5_model.h
    src/t5_model.cpp
)

# Include directories
target_include_directories(island_npc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT}/include
    ${sentencepiece_SOURCE_DIR}/src    # Add SentencePiece headers
)

# Link against ONNX Runtime and SentencePiece
target_link_libraries(island_npc PRIVATE
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib
    sentencepiece-static    # Use static library
)

# Copy ONNX model and tokenizer config to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models
     DESTINATION ${CMAKE_BINARY_DIR}
     PATTERN "*")

# Add test executables
add_executable(tokenizer_test
    tests/tokenizer_test.cpp
    src/tokenizer.h
    src/tokenizer.cpp
)

add_executable(t5_model_test
    tests/t5_model_test.cpp
    src/t5_model.h
    src/t5_model.cpp
    src/tokenizer.h
    src/tokenizer.cpp
)

# Configure test includes
foreach(test_target tokenizer_test t5_model_test)
    target_include_directories(${test_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${munit_SOURCE_DIR}/
        ${sentencepiece_SOURCE_DIR}/src
        ${ONNXRUNTIME_ROOT}/include
    )

    target_include_directories(${test_target} PRIVATE
        ${CMAKE_BINARY_DIR}
    )
endforeach()

# Copy munit header to build directory
configure_file(
    ${munit_SOURCE_DIR}/munit.h
    ${CMAKE_BINARY_DIR}/munit.h
    COPYONLY
)

# Link test dependencies
target_link_libraries(tokenizer_test PRIVATE
    munit
    sentencepiece-static
)

target_link_libraries(t5_model_test PRIVATE
    munit
    sentencepiece-static
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib
)

enable_testing()
add_test(NAME tokenizer_test COMMAND tokenizer_test)
add_test(NAME t5_model_test COMMAND t5_model_test --model ${CMAKE_BINARY_DIR}/models/t5_tiny.onnx --sp_model ${CMAKE_BINARY_DIR}/models/spiece.model)
