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
    int unk_id = tokenizer.GetUnkTokenId();
    
    munit_assert_int(3, ==, unk_id);
    
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

static MunitResult test_unrecognized_tokens(const MunitParameter params[], void* user_data) {
    Tokenizer tokenizer(model_path);

    printf("Test unrecognized tokens\n");
    std::vector<std::string> tokens = {"brillig", "slithy", "tove"};
    // for brillig the expectedd tokenization is the unk token, bri, l, and lig.
    std::vector<int64_t> expected_tokens[3] =
        {{tokenizer.GetUnkTokenId(), 2160, 40, 2825}, // unk, bri, l, lig
         {tokenizer.GetUnkTokenId(), 7, 18800, 63}, // unk, s, lith, y
        {12, 162}}; // to, ve ~ nb. for some reason "tove" is recognized as two tokens, with no unk tag.
    int expected = 0;
    for (const auto& token : tokens) {
        auto token_ids = tokenizer.tokenize({token});
        #if 0
        for (auto id : token_ids) {
            printf("Token: %s, ID: %lld\n", token.c_str(), (long long)id);
        }
        std::string result = tokenizer.detokenize(token_ids);
        printf("Detokenized: %s\n", result.c_str());
        for (auto id : token_ids) {
            std::string token = tokenizer.detokenize({id});
            printf("Detokenized ID: %lld, Token: %s\n", (long long)id, token.c_str());
        }
        #endif
        // munit assert the expected tokens match.
        munit_assert(token_ids.size() == expected_tokens[expected].size());
        for (size_t i = 0; i < token_ids.size(); ++i) {
            munit_assert_int64(token_ids[i], ==, expected_tokens[expected][i]);
        }
        ++expected;
    }

    return MUNIT_OK;
}


static MunitResult test_jabberwocky_tokenization(const MunitParameter params[], void* user_data) {
    Tokenizer tokenizer(model_path);

    printf("Jabberwocky test\n");
    std::string jabberwocky = "Twas brillig, and the slithy toves did gyre and gimble in the wabe.";
    std::vector<int64_t> expected_tokens = {
        332, // T
        9491, // was
        tokenizer.GetUnkTokenId(), 2160, 40, 2825, // brillig
        6, // ,
        11, // and
        8, // the
        tokenizer.GetUnkTokenId(), 7, 18800, 63, // slithy
        12, 162, 7, // toves
        410, // did
        tokenizer.GetUnkTokenId(), 122, 63, 60, // gyre
        11, // and
        tokenizer.GetUnkTokenId(), 122, 603, 2296, // gimble
        16, // in
        8, // the
        8036, 346, // wabe
        5, // . 
    };

    std::vector<int64_t> token_ids = tokenizer.tokenize(jabberwocky);
    std::string check_jabberwocky = tokenizer.detokenize(token_ids);

    munit_assert(jabberwocky == check_jabberwocky);

#if 0
    for (auto id : token_ids) {
        std::string token = tokenizer.detokenize({id});
        printf("Detokenized ID: %lld, Token: %s\n", (long long)id, token.c_str());
    }

    for (size_t i = 0; i < expected_tokens.size(); ++i) {
        printf("Expected: %lld, Actual: %lld\n", (long long)expected_tokens[i], (long long)token_ids[i]);
        munit_assert_int64(token_ids[i], ==, expected_tokens[i]);
    }
#endif
    return MUNIT_OK;
}

static MunitTest tests[] = {
    { (char*)"/special_tokens",
        test_special_tokens,
        test_setup, test_teardown,
        MUNIT_TEST_OPTION_NONE, NULL
    },
    { (char*)"/tokenize_detokenize",
        test_tokenize_detokenize,
        test_setup, test_teardown,
        MUNIT_TEST_OPTION_NONE, NULL
    },
    { (char*) "/test_unrecognized_tokens", 
        test_unrecognized_tokens, 
        test_setup, test_teardown,
         MUNIT_TEST_OPTION_NONE, NULL 
    },
    { (char*) "/test_jabberwocky_tokenization", 
       test_jabberwocky_tokenization, 
       test_setup, test_teardown,
       MUNIT_TEST_OPTION_NONE, NULL },

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
