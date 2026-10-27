#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target) {
    const auto h = config.hidden_size;
    MtpWeights out;
    out.input_projection = b.parameter("mtp/input_projection", {h, 2ULL * h}, {"mtp/stem_input"});
    out.embedding_norm   = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm      = b.direct("mtp/hidden_norm", {h});
    out.final_norm       = b.direct("mtp/final_norm", {h});
    out.layer            = bind_block(b, config, "mtp/layers/0/", MixerKind::FullAttention);
    out.token_embedding  = target.token_embedding;
    // A dedicated draft table is optional: the artifact either stores one (`mtp/token_embedding`,
    // Qwen's `mtp.embeddings`) or the head shares the text table. Presence in the artifact decides,
    // so no config key has to travel with it and older artifacts keep loading.
    if (b.binder.contains("mtp/token_embedding")) {
        out.own_token_embedding =
            b.parameter("mtp/token_embedding", {config.vocab_size, h});
    }
    out.output_head      = target.output_head;
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
