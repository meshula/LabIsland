cmake_minimum_required(VERSION 3.12)
project(island_npc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

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

# Add ONNX Runtime include and library directories
include_directories("${ONNXRUNTIME_ROOT}/include")
link_directories("${ONNXRUNTIME_ROOT}/lib")

# Add executable
add_executable(island_npc 
    src/main.cpp
    src/npc.cpp
)

# Include directories
target_include_directories(island_npc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against ONNX Runtime
target_link_libraries(island_npc PRIVATE
    onnxruntime
)

# Copy ONNX model and tokenizer config to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models
     DESTINATION ${CMAKE_BINARY_DIR}
     PATTERN "*")
