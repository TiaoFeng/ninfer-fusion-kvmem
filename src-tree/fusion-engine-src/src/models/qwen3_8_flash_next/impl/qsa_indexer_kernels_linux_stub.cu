// Linux build: the Qwen4-preview QSA indexer is the only unit in this tree that needs
// <cub/device/device_topk.cuh>. The CCCL bundled with CUDA 13.1 does not ship DeviceTopK
// (cub/device/device_segmented_radix_sort.cuh is there, device_topk.cuh is not), and pulling a
// newer CCCL would mean fetching third-party code at build time.
//
// This file replaces `qsa_indexer_kernels.cu` in `runtime_sources.cmake` for that reason. The
// qwen3_5 line -- which is what the Bonsai 2 27B artifact uses -- never reaches these entry points,
// and a caller that does reach them fails loudly rather than computing a wrong selection.

#include "models/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"

#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

[[noreturn]] void qsa_indexer_unavailable() {
    throw std::logic_error(
        "flash_next QSA indexer kernels are not compiled into this Linux engine: the CUDA 13.1 "
        "CCCL has no cub/device/device_topk.cuh");
}

} // namespace

void flash_next_qsa_indexer_store_launch(const Tensor& projected, const Tensor& token_indices,
                                         const Tensor& mrope_positions, const Tensor& table_rows,
                                         const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots,
                                         const Tensor& key_norm, QsaIndexerCacheView cache,
                                         cudaStream_t stream, bool aliased_recurrent_scan) {
    (void)projected;
    (void)token_indices;
    (void)mrope_positions;
    (void)table_rows;
    (void)source_state_slots;
    (void)destination_state_slots;
    (void)key_norm;
    (void)cache;
    (void)stream;
    (void)aliased_recurrent_scan;
    qsa_indexer_unavailable();
}

void flash_next_qsa_indexer_store_prefill_launch(
    const Tensor& projected, const Tensor& token_indices, const Tensor& mrope_positions,
    std::int32_t table_row, std::int32_t source_state_slot, std::int32_t destination_state_slot,
    const Tensor& key_norm, QsaIndexerCacheView cache, cudaStream_t stream) {
    (void)projected;
    (void)token_indices;
    (void)mrope_positions;
    (void)table_row;
    (void)source_state_slot;
    (void)destination_state_slot;
    (void)key_norm;
    (void)cache;
    (void)stream;
    qsa_indexer_unavailable();
}

std::size_t flash_next_qsa_indexer_sort_temp_bytes(std::int32_t maximum_blocks, std::int32_t batch) {
    (void)maximum_blocks;
    (void)batch;
    qsa_indexer_unavailable();
}

void flash_next_qsa_indexer_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                   const Tensor& table_rows, const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, const Tensor& query_norm,
                                   const Tensor& key_norm, QsaIndexerCacheView cache,
                                   FlashNextQsaIndexerWorkspace& scratch, std::int32_t active_blocks,
                                   Tensor& selected_blocks, Tensor& selected_counts,
                                   cudaStream_t stream, bool aliased_recurrent_scan) {
    (void)token_indices;
    (void)mrope_positions;
    (void)table_rows;
    (void)source_state_slots;
    (void)destination_state_slots;
    (void)query_norm;
    (void)key_norm;
    (void)cache;
    (void)scratch;
    (void)active_blocks;
    (void)selected_blocks;
    (void)selected_counts;
    (void)stream;
    (void)aliased_recurrent_scan;
    qsa_indexer_unavailable();
}

void flash_next_qsa_indexer_prefill_launch(
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    std::int32_t source_state_slot, std::int32_t destination_state_slot, const Tensor& query_norm,
    const Tensor& key_norm, QsaIndexerCacheView cache, FlashNextQsaIndexerWorkspace& scratch,
    std::int32_t maximum_blocks, std::int32_t first_token_index, Tensor& selected_blocks,
    Tensor& selected_counts, cudaStream_t stream) {
    (void)token_indices;
    (void)mrope_positions;
    (void)table_row;
    (void)source_state_slot;
    (void)destination_state_slot;
    (void)query_norm;
    (void)key_norm;
    (void)cache;
    (void)scratch;
    (void)maximum_blocks;
    (void)first_token_index;
    (void)selected_blocks;
    (void)selected_counts;
    (void)stream;
    qsa_indexer_unavailable();
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
