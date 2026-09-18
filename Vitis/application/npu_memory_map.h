/*
 * LeeNet11 NPU - C-side DDR memory map (proposal v1)
 *
 * Source of truth for sizes: A-to-C Model/Memory ABI Spec (Docs/LeeNet11-NPU-INT16-v1_A-to-C_Model_Memory_ABI_Spec.docx)
 * Address layout decisions: Docs/C_DDR_Memory_Map_v1.md
 *
 * Not yet cross-checked against a generated Vitis lscript.ld (no Vitis project exists yet).
 * APP region upper bound (NPU_APP_REGION_END) must be re-verified once the standalone
 * platform is created, so heap/stack cannot grow past it into NPU_MODEL_BASE.
 */

#ifndef NPU_MEMORY_MAP_H
#define NPU_MEMORY_MAP_H

#include <stdint.h>

/* AXI4-Lite register aperture; fixed by the v3 Vivado address map. */
#define NPU_REG_BASE           0x43C00000u

/* ---- Region bases (all 1MB-aligned) ---- */
#define NPU_APP_BASE           0x00000000u
#define NPU_APP_REGION_END     0x00FFFFFFu   /* inclusive; app .text/.data/.bss/heap/stack must stay <= this */

#define NPU_MODEL_BASE         0x01000000u
#define NPU_MODEL_REGION_SIZE  0x00200000u   /* 2 MB slot; actual model_params.bin = 1,509,455 B */

#define NPU_INPUT_BASE         0x01200000u
#define NPU_INPUT_REGION_SIZE  0x00100000u   /* 1 MB slot; actual input = 640,000 B (320000 x INT16) */

#define NPU_OUTPUT_BASE        0x01300000u
#define NPU_OUTPUT_REGION_SIZE 0x00100000u   /* 1 MB slot; actual output = 1,054 B (527 x INT16 Q10) */

#define NPU_ACT_A_BASE         0x01400000u
#define NPU_ACT_B_BASE         0x02400000u
#define NPU_ACT_REGION_SIZE    0x01000000u   /* 16 MB slot each; actual max tensor (Conv1 out) = 13,653,376 B */

#define NPU_GOLDEN_BASE        0x03400000u   /* optional, integration/debug only */
#define NPU_GOLDEN_REGION_SIZE 0x02000000u   /* 32 MB slot */

/* ---- Exact ABI sizes (for bounds-checking driver code, not for base-address math) ---- */
#define NPU_MODEL_PARAMS_BYTES   1509455u
#define NPU_INPUT_BYTES          640000u
#define NPU_OUTPUT_BYTES         1054u
#define NPU_ACT_MAX_TENSOR_BYTES 13653376u   /* Conv1 output [64][106667] INT16 */

/* ---- Ping-pong helper ----
 * Conv1..Conv9 alternate ACT_A/ACT_B by layer parity, ending on ACT_A after Conv9 (9 layers, odd count).
 * layer_idx: 1-based conv layer index (Conv1=1 .. Conv9=9).
 * Returns the *output* buffer base for that conv layer.
 */
static inline uint32_t npu_conv_output_base(unsigned layer_idx) {
    return (layer_idx % 2u) ? NPU_ACT_A_BASE : NPU_ACT_B_BASE;
}

#endif /* NPU_MEMORY_MAP_H */
