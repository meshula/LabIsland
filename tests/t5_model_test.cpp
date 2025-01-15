#include "../src/t5_model.h"
#include "munit.h"
#include <string>

// Test helper class that exposes protected members
class TestT5Model : public T5Model {
public:
    using T5Model::T5Model;
    using T5Model::tokenizer_;
    using T5Model::tokens_to_tensors;
    using T5Model::tensor_to_next_token;
};

static char* model_path = NULL;
static char* sp_model_path = NULL;

static void* test_setup(const MunitParameter params[], void* user_data) {
    (void)params;
    (void)user_data;
    return NULL;
}

static void test_teardown(void* fixture) {
    (void)fixture;
}

static MunitResult test_tokenization(const MunitParameter params[], void* fixture) {
    (void)params;
    (void)fixture;

    // Create TestT5Model instance
    TestT5Model model(model_path, sp_model_path);

    // First paragraph of 1984
    const std::string input_text = 
        "It was a bright cold day in April, and the clocks were striking thirteen.";

    // Tokenize input text
    std::vector<int64_t> tokens = model.tokenizer_.tokenize(input_text);
    munit_assert_size(tokens.size(), >, 0);

    // Print tokens for debugging
    printf("Tokens: ");
    for (auto token : tokens) {
        printf("%lld ", (long long)token);
    }
    printf("\n");

    // Verify we can detokenize back
    std::string result = model.tokenizer_.detokenize(tokens);
    munit_assert_string_equal(result.c_str(), input_text.c_str());

    return MUNIT_OK;
}

static MunitResult test_tensor_creation(const MunitParameter params[], void* fixture) {
    (void)params;
    (void)fixture;

    // Create TestT5Model instance
    TestT5Model model(model_path, sp_model_path);

    // Simple test input
    const std::string input_text = "Hello world.";
    std::vector<int64_t> tokens = model.tokenizer_.tokenize(input_text);
    
    printf("Creating tensors for tokens: ");
    for (auto token : tokens) {
        printf("%lld ", (long long)token);
    }
    printf("\n");

    // Convert tokens to tensors
    auto input_tensors = model.tokens_to_tensors(tokens);
    munit_assert_size(input_tensors.size(), ==, 4);

    // Verify tensor shapes
    for (size_t i = 0; i < input_tensors.size(); i++) {
        auto shape = input_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        printf("Tensor %zu shape: [%lld, %lld]\n", i, 
               (long long)shape[0], (long long)shape[1]);
        munit_assert_size(shape.size(), ==, 2);
        munit_assert_int64(shape[0], ==, 1);  // batch size
    }

    return MUNIT_OK;
}

static MunitResult test_tensor_to_token(const MunitParameter params[], void* fixture) {
    (void)params;
    (void)fixture;

    // Create TestT5Model instance
    TestT5Model model(model_path, sp_model_path);

    // Create a mock output tensor with known logits
    // Shape: [batch_size=1, sequence_length=1, vocab_size=4]
    std::vector<int64_t> shape = {1, 1, 4};
    std::vector<float> logits = {-1.0f, 2.0f, 0.5f, -0.5f};  // Second token should have highest probability
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, logits.data(), logits.size(), shape.data(), shape.size());

    // Test token prediction
    int64_t predicted_token = model.tensor_to_next_token(output_tensor, 0);
    printf("Predicted token from mock logits: %lld\n", (long long)predicted_token);
    
    // Should predict token 1 (highest logit value)
    munit_assert_int64(predicted_token, ==, 1);

    return MUNIT_OK;
}

static MunitTest tests[] = {
    {
        (char*)"/tokenization",
        test_tokenization,
        test_setup,
        test_teardown,
        MUNIT_TEST_OPTION_NONE,
        NULL
    },
    {
        (char*)"/tensor_creation",
        test_tensor_creation,
        test_setup,
        test_teardown,
        MUNIT_TEST_OPTION_NONE,
        NULL
    },
    {
        (char*)"/tensor_to_token",
        test_tensor_to_token,
        test_setup,
        test_teardown,
        MUNIT_TEST_OPTION_NONE,
        NULL
    },
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

static const MunitSuite suite = {
    (char*)"/t5_model",
    tests,
    NULL,
    1,
    MUNIT_SUITE_OPTION_NONE
};

int main(int argc, char* argv[]) {
    printf("T5 Model test\n");
    if (argc < 5) {
        fprintf(stderr, "Usage: %s --model <model_path> --sp_model <sp_model_path>\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
            // Remove these arguments
            for (int j = i; j < argc - 2; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--;
        }
        else if (strcmp(argv[i], "--sp_model") == 0 && i + 1 < argc) {
            sp_model_path = argv[i + 1];
            // Remove these arguments
            for (int j = i; j < argc - 2; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--;
        }
    }

    if (!model_path || !sp_model_path) {
        fprintf(stderr, "Both --model and --sp_model arguments are required\n");
        return 1;
    }

    printf("Model path: %s\n", model_path);
    printf("SentencePiece model path: %s\n", sp_model_path);

    return munit_suite_main(&suite, NULL, argc, argv);
}
