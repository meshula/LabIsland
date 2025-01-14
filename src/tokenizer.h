#include <sentencepiece_processor.h>
#include <vector>
#include <string>
#include <iostream>

class Tokenizer {
public:
    Tokenizer(const std::string& sp_model_path) {
        if (!sp.Load(sp_model_path).ok()) {
            throw std::runtime_error("Failed to load SentencePiece model.");
        }
    }

    std::vector<int64_t> tokenize(const std::string& input) {
        std::vector<int> piece_ids;
        sp.Encode(input, &piece_ids);
        return std::vector<int64_t>(piece_ids.begin(), piece_ids.end());
    }

    std::string detokenize(const std::vector<int64_t>& tokens) {
        std::vector<int> int_tokens(tokens.begin(), tokens.end());
        std::string result;
        if (sp.Decode(int_tokens, &result).ok())
            return result;
        return "";
    }

    // Get special token IDs
    int GetPadTokenId() const {
        std::vector<int> ids;
        auto status = sp.Encode("<pad>", &ids);
        return status.ok() && !ids.empty() ? ids[0] : 0;
    }

    int GetEosTokenId() const {
        std::vector<int> ids;
        auto status = sp.Encode("</s>", &ids);
        return status.ok() && !ids.empty() ? ids[0] : 1;
    }

    int GetUnkTokenId() const {
        std::vector<int> ids;
        auto status = sp.Encode("<unk>", &ids);
        return status.ok() && !ids.empty() ? ids[0] : 2;
    }

    size_t GetVocabSize() const {
        return sp.GetPieceSize();
    }

private:
    sentencepiece::SentencePieceProcessor sp;
};
