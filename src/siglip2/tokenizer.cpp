#include "siglip2/siglip2.h"

#include "sentencepiece_processor.h"

#include <algorithm>

namespace siglip2 {

struct Tokenizer::State {
    sentencepiece::SentencePieceProcessor proc;
};

Tokenizer::Tokenizer() = default;

Tokenizer::~Tokenizer() {
    close();
}

void Tokenizer::close() {
    delete state_;
    state_ = nullptr;
}

bool Tokenizer::load(std::string const& spm_model_path) {
    close();
    auto st = std::make_unique<State>();
    auto status = st->proc.Load(spm_model_path);
    if (!status.ok()) {
        error_msg_ = "sentencepiece Load failed: " + status.ToString();
        return false;
    }
    pad_id_ = st->proc.pad_id();
    eos_id_ = st->proc.eos_id();
    if (pad_id_ < 0) {
        // SigLIP2/Gemma sentencepiece models leave pad_id undefined; the
        // convention is pad_token_id=0.
        pad_id_ = 0;
    }
    state_ = st.release();
    return true;
}

bool Tokenizer::encode(
    std::string const& text,
    int max_length,
    std::vector<int32_t>& out_token_ids,
    std::vector<int32_t>& out_attention_mask) {
    if (!state_) {
        error_msg_ = "Tokenizer not loaded";
        return false;
    }
    std::vector<int> ids;
    auto status = state_->proc.Encode(text, &ids);
    if (!status.ok()) {
        error_msg_ = "Encode failed: " + status.ToString();
        return false;
    }

    // Plain sentencepiece Encode does not append EOS; SigLIP2/Gemma expects it.
    if (eos_id_ >= 0) {
        ids.push_back(eos_id_);
    }

    if (max_length <= 0) {
        out_token_ids.assign(ids.begin(), ids.end());
        out_attention_mask.assign(ids.size(), 1);
        return true;
    }

    out_token_ids.assign((size_t)max_length, pad_id_);
    out_attention_mask.assign((size_t)max_length, 0);

    int const n_keep = std::min((int)ids.size(), max_length);
    for (int i = 0; i < n_keep; ++i) {
        out_token_ids[i] = ids[i];
        out_attention_mask[i] = 1;
    }
    if (eos_id_ >= 0 && (int)ids.size() > max_length) {
        out_token_ids[max_length - 1] = eos_id_;
    }
    return true;
}

} // namespace siglip2
