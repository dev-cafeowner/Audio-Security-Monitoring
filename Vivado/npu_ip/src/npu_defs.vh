`ifndef NPU_DEFS_VH
`define NPU_DEFS_VH
//////////////////////////////////////////////////////////////////////////////
// LeeNet11 NPU - shared definitions for the C-side wrapper (dummy_npu_core.v
// and friends) that drives b_compute_top_16.
//
// Source of truth:
//   - b_compute_top_16.v port/parameter list (this file mirrors its defaults)
//   - Docs/B_to_C_Compute_Interface_Spec_v1_FINAL_FROZEN.md (FINAL FROZEN)
//////////////////////////////////////////////////////////////////////////////

// ---- Operand / accumulator widths (mirrors b_compute_top_16 defaults) ----
`define NPU_DATA_W          16   // activation, INT16
`define NPU_WEIGHT_W        16   // weight, INT16
`define NPU_ACC_W           48   // bias / accumulator, signed48
`define NPU_OUT_W           16   // unified output, INT16
`define NPU_SHIFT_W          8   // per-lane rshift container

// ---- Config field widths (mirrors b_compute_top_16 defaults) ----
`define NPU_CFG_C_W         10   // conv Cin/Cout width
`define NPU_CFG_T_W         20   // conv time-index width
`define NPU_CFG_DIM_W       10   // FC in/out dim width
`define NPU_GROUP_W          6   // result/output group index width
`define NPU_MAX_CONV_GROUPS 16
`define NPU_GLOBAL_GROUP_W   4
`define NPU_GLOBAL_TIME_W    5

// ---- Derived parallel bus widths (PE-side, one accepted compute step) ----
`define NPU_LANES           16
`define NPU_TAPS             3
`define NPU_PIN              4
`define NPU_WEIGHT_BUS_W    (`NPU_LANES * `NPU_TAPS * `NPU_WEIGHT_W)  // 768
`define NPU_PIN4_X_BUS_W    (`NPU_PIN * `NPU_TAPS * `NPU_DATA_W)      // 192
`define NPU_PIN4_WEIGHT_BUS_W (`NPU_LANES * `NPU_PIN * `NPU_TAPS * `NPU_WEIGHT_W) // 3072
`define NPU_BIAS_BUS_W      (`NPU_LANES * `NPU_ACC_W)                 // 768
`define NPU_RSHIFT_BUS_W    (`NPU_LANES * `NPU_SHIFT_W)               // 128
`define NPU_GLOBAL_BUS_W    (`NPU_LANES * `NPU_DATA_W)                // 256
`define NPU_OUT_BUS_W       (`NPU_LANES * `NPU_OUT_W)                 // 256

// ---- i_op_mode encoding (B_to_C spec S1) ----
`define NPU_OP_CONV    2'b00
`define NPU_OP_FC      2'b01
`define NPU_OP_GLOBAL  2'b10
// 2'b11 is reserved - b_compute_top_16 flags it as an error (never assert i_start with this value)

// ---- 768-bit weight bus packing (B_to_C spec S12, FINAL FROZEN) ----
// lane-major / tap-minor: word_index = lane*3 + tap, i_weight[(word_index*16) +: 16]
// tap0 -> i_x0, tap1 -> i_x1, tap2 -> i_x2
`define NPU_WEIGHT_BIT_OFS(lane, tap) (((lane) * `NPU_TAPS + (tap)) * `NPU_WEIGHT_W)
`define NPU_PIN4_X_BIT_OFS(pin, tap) (((pin) * `NPU_TAPS + (tap)) * `NPU_DATA_W)
`define NPU_PIN4_WEIGHT_BIT_OFS(lane, pin, tap) \
    ((((lane) * `NPU_PIN * `NPU_TAPS) + ((pin) * `NPU_TAPS) + (tap)) * `NPU_WEIGHT_W)

// ---- Bias / Rshift lane packing (B_to_C spec S5) ----
`define NPU_BIAS_BIT_OFS(lane)   ((lane) * `NPU_ACC_W)
`define NPU_RSHIFT_BIT_OFS(lane) ((lane) * `NPU_SHIFT_W)

// ---- GLOBAL interface shape (B_to_C spec S6, FINAL FROZEN) ----
// 256 channels x 17 time, 16 channels/beat -> 16 groups x 17 times = 272 accepted beats
// order = Time outer / OC-group inner; group 0..15 for time 0, then time 1, ... time 16
`define NPU_GLOBAL_LAST_GROUP 4'd15
`define NPU_GLOBAL_LAST_TIME  5'd16   // 0-indexed, so 17 time steps total (0..16)
`define NPU_GLOBAL_NUM_GROUPS 16
`define NPU_GLOBAL_NUM_TIMES  17
`define NPU_GLOBAL_TOTAL_BEATS 272

// ---- Unified output contract (B_to_C spec S7) ----
`define NPU_OUT_LANE_BIT_OFS(lane) ((lane) * `NPU_OUT_W)
`define NPU_FC2_LAST_GROUP_MASK 16'h7FFF   // OC512..526, 15 valid lanes

// Per-layer Cin/Cout/OutLen/FCIn/FCOut/AcceptedComputeSteps tables live in
// Docs/B_to_C_Layer_Config_v1_FINAL_FROZEN.csv, not here - this file only
// carries the fixed structural constants shared by every layer/mode.

`endif // NPU_DEFS_VH
