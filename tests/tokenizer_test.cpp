#include "../src/tokenizer.h"
#include "munit.h"
#include <string>

static char* model_path = NULL;

static void* test_setup(const MunitParameter params[], void* user_data) {
    (void)params;
    (void)user_data;
    return NULL;
}

static void test_teardown(void* fixture) {
    (void)fixture;
}

// Test special token IDs
static MunitResult test_special_tokens(const MunitParameter params[], void* fixture) {
    (void)params;
    (void)fixture;
    
    Tokenizer tokenizer(model_path);
    
    // Get special token IDs
    int pad_id = tokenizer.GetPadTokenId();
    int eos_id = tokenizer.GetEosTokenId();
    int unk_id = tokenizer.GetUnkTokenId();
    
    // Print token IDs for debugging
    printf("pad_id: %d, eos_id: %d, unk_id: %d\n", pad_id, eos_id, unk_id);
    
    // Verify they are different and have expected values
    munit_assert_int(pad_id, !=, eos_id);
    munit_assert_int(pad_id, !=, unk_id);
    munit_assert_int(eos_id, !=, unk_id);
    
    return MUNIT_OK;
}

// Test tokenization and detokenization
static MunitResult test_tokenize_detokenize(const MunitParameter params[], void* fixture) {
    (void)params;
    (void)fixture;
    
    Tokenizer tokenizer(model_path);
    std::string test_text = "Hello world! This is a test.";
    
    // Tokenize
    auto tokens = tokenizer.tokenize(test_text);
    
    // Print tokens for debugging
    printf("Tokens: ");
    for (auto token : tokens) {
        printf("%lld ", (long long)token);
    }
    printf("\n");
    
    // Detokenize
    std::string result = tokenizer.detokenize(tokens);
    
    // Print result for debugging
    printf("Original: '%s'\n", test_text.c_str());
    printf("Result:   '%s'\n", result.c_str());
    
    // Verify the result matches (allowing for some whitespace differences)
    munit_assert_string_equal(result.c_str(), test_text.c_str());
    
    return MUNIT_OK;
}
static MunitTest tests[] = {
    {
        (char*)"/special_tokens",
        test_special_tokens,
        test_setup,
        test_teardown,
        MUNIT_TEST_OPTION_NONE,
        NULL
    },
    {
        (char*)"/tokenize_detokenize",
        test_tokenize_detokenize,
        test_setup,
        test_teardown,
        MUNIT_TEST_OPTION_NONE,
        NULL
    },
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};


static const MunitSuite suite = {
    (char*)"/tokenizer",
    tests,
    NULL,
    0,  // Changed from 1 to 0 to not skip tests
    MUNIT_SUITE_OPTION_NONE
};

int main(int argc, char* argv[]) {
    printf("Tokenizer test\n");
    if (argc < 3) {
        fprintf(stderr, "Usage: %s --model <model_path>\n", argv[0]);
        return 1;
    }

    // Filter custom arguments
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--model", 8) == 0) {
            model_path = argv[i] + 8;  // Extract model path after "--model="
            // Remove custom argument from argv
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            argc -= 2;

            printf("Model path: %s\n", model_path);
            break;
        }
    }
    
    if (model_path == NULL) {
        fprintf(stderr, "Usage: %s --model=<model_path>\n", argv[0]);
        return 1;
    }

    // print arguments
    for (int i = 0; i < argc; ++i) {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    return munit_suite_main(&suite, NULL, argc, argv);
}
