`timescale 1ns / 1ps
`include "npu_defs.vh"

//////////////////////////////////////////////////////////////////////////////
// npu_conv_adapter_cin_n - synthetic, reduced-parameter Cin>1 CONV adapter
// (Docs/C_Model_ABI_and_Scheduler_Design_v1.md S1.1.1/S1.2, step 1 of the
// Conv2 verification order agreed with the reviewer). This is NOT a new
// packaged IP and does NOT touch neural_processing_unit_conv1_1_0 or its CSR
// - it is a bare RTL module driven directly by tb_conv_adapter_cin_n.v via
// xsim, scoped to prove the line-buffer / weight-bank / single-AXIS-LOAD
// design before it is wired into any real descriptor/CSR path.
//
// Backend handshake contract is IDENTICAL to npu_conv1_backend_adapter.v
// (o_error_pulse/o_error_code[7:0]/i_soft_reset_pulse, same b_rst_n_reg /
// adapter_clear / op_active pattern, error-resets-B-core-too) - reviewer
// requirement: reuse the real Conv1/CSR backend contract, not a bespoke one.
//
// LOAD is a SINGLE sequential S_AXIS stream: weight -> bias -> rshift ->
// activation, exactly like Conv1 (real SoC has one MM2S channel; splitting
// LOAD into separate AXIS ports would stop this from exercising the real
// DMA/FSM sequencing and would not be reusable by a future unified adapter).
//
// Generalizations vs npu_conv1_backend_adapter.v (Cin=1 special case):
//   - i_cin/i_num_oc_group/i_conv_out_len/i_pool_enable are RUNTIME inputs,
//     latched at an accepted START (descriptor bounds, design doc S4), not
//     module parameters. i_num_oc_group range 1..16 and i_cin range
//     1..MAX_CIN are checked BEFORE accepting START (design doc S4.2 style -
//     invalid config rejects START itself, code ERR_CFG_INVALID=0x16,
//     mirrors the CSR-level LAYER_ID rejection mechanism at adapter scope).
//   - Weight local buffer is the 16-bank x 48-bit x 2048-deep structure from
//     design doc S1.1.1 (addr = oc_group*128 + cin, fixed 128 stride
//     regardless of actual i_cin - "일부만 사용" addressing), not Conv1's
//     flat per-group 768-bit register.
//   - Activation is a four-row line buffer: three physical rows form the
//     active K=3 window and the fourth is filled ahead of the next time
//     step.  The mod-4 fill/read pointers avoid a refill bubble while keeping
//     the same three-tap convolution semantics - no data
//     shifting, no division. i_x0/i_x1/i_x2 are combinational reads keyed by
//     B's own o_conv_cin_idx (issue-side, trusted - same trust model Conv1
//     already applies to o_conv_oc_group_idx), reused across every
//     (oc_group, cin) step of the current time window without re-fetch.
//     K=3/S=1/P=1 (same-padding): t=0 forces i_x0=0, t=out_len-1 forces
//     i_x2=0 (see act_row_valid/first_window/last_window below).
//   - Output-side "last group" detection still does NOT trust any B
//     result-side signal (design doc S1.3 principle unchanged) - it is the
//     adapter's own output_group_count against last_output_group_r, a
//     manifest-authoritative value (i_last_output_group) latched at START,
//     not computed by this adapter (see i_conv_post_out_len/
//     i_last_output_group port comments - both `/3` and the output-group
//     multiply were found to violate OOC timing regardless of which
//     register captured the result, so neither is computed on-chip any
//     more).
//
// Scope restriction for this step (reviewer P2, source review round 2): only
// EVEN i_cin and i_conv_out_len!=0 are accepted (rejected otherwise via the
// existing 0x16 path) - the RUN-phase fill logic always pushes both
// halfwords of a beat unconditionally, so it has no partial-beat path for an
// odd Cin, and the last-index arithmetic below underflows at
// i_conv_out_len==0. Real Conv2-9 (Cin=64 or 128) are unaffected; this is a
// documented gap in the adapter's generality, not a limitation on the
// intended target.
//
// Debug ports (dbg_*) expose the adapter's own issue-side tracking
// (time/oc_group/cin/step_fire) for the testbench's input-window scoreboard
// - reviewer requirement: check taps at every accepted step, not just final
// output, so line-buffer bugs are localized instead of only showing up as a
// wrong final result.
//////////////////////////////////////////////////////////////////////////////

// Conv2~9 frontend for neural_processing_unit_unified_2_0.
// Derived from the frozen poolinfix adapter.  Its LOAD/RUN/DRAIN FSMs and
// local storage are unchanged; only the private B-core instance is replaced
// by the explicit shared-core boundary below.
module npu_convn_frontend #(
    parameter MAX_CIN      = 128,
    parameter MAX_OC_GROUP = 16,
    parameter USE_SHARED_PARAMS = 0
) (
    input  wire        clk,
    input  wire        rst_n,

    // ---- backend handshake (identical contract to npu_conv1_backend_adapter.v) ----
    input  wire        i_start_pulse,
    input  wire        i_soft_reset_pulse,
    output wire        o_busy,
    output wire        o_done,
    output wire        o_error_pulse,
    output wire [7:0]  o_error_code,

    // ---- layer descriptor bounds (sampled at an accepted START) ----
    input  wire [`NPU_CFG_C_W-1:0] i_cin,          // 1..MAX_CIN
    input  wire [4:0]              i_num_oc_group, // 1..MAX_OC_GROUP
    input  wire [`NPU_CFG_T_W-1:0] i_conv_out_len, // Tconv (pre-pool)
    input  wire                    i_pool_enable,
    // Manifest-authoritative post-pool geometry (Docs/C_to_B_ConvOutLen_
    // ConfigTiming_Review_Request_v1.md follow-up decision: NOT computed by
    // this adapter any more - `/3` and the output-group multiply are both
    // expensive combinational operations that already violated OOC timing
    // once each (see CFG_DERIVE comment below and its removal). The caller
    // (fixed Conv2/Conv9-style wrapper tie-off today, a future layer-
    // descriptor ROM later) already knows out_len and last_output_group from
    // the same manifest that sets i_conv_out_len/i_pool_enable, so supplying
    // them directly here removes the division/multiply from the adapter's
    // and B's START/configuration path entirely instead of re-pipelining it.
    input  wire [`NPU_CFG_T_W-1:0] i_conv_post_out_len, // out_len (post-pool)
    input  wire [31:0]             i_last_output_group, // num_oc_group*out_len-1
    input  wire [31:0]             i_activation_beats,  // exact Tconv*Cin/2

    // ---- S_AXIS: single sequential LOAD (weight->bias->rshift) then RUN (activation) ----
    output wire        s_axis_tready,
    input  wire [31:0] s_axis_tdata,
    input  wire [3:0]  s_axis_tkeep,
    input  wire        s_axis_tlast,
    input  wire        s_axis_tvalid,

    // ---- M_AXIS: result stream (TM_B, [Time][Channel]) ----
    output wire        m_axis_tvalid,
    output wire [31:0] m_axis_tdata,
    output wire [3:0]  m_axis_tkeep,
    output wire        m_axis_tlast,
    input  wire        m_axis_tready,

    // ---- shared B-core reset/control ----
    output wire                      o_b_reset_request,
    output wire                      o_b_start,
    output wire                      o_b_pool_enable,
    output wire [`NPU_CFG_C_W-1:0]   o_b_conv_cin,
    output wire [`NPU_CFG_C_W-1:0]   o_b_conv_cout,
    output wire [`NPU_CFG_T_W-1:0]   o_b_conv_out_len,
    output wire [`NPU_CFG_T_W-1:0]   o_b_conv_post_out_len,

    // ---- shared B-core operand ingress ----
    output wire                       o_b_act_valid,
    input  wire                       i_b_act_ready,
    output wire [`NPU_PIN4_X_BUS_W-1:0] o_b_x,
    output wire                       o_b_weight_valid,
    input  wire                       i_b_weight_ready,
    output wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] o_b_weight,
    output wire [`NPU_BIAS_BUS_W-1:0]   o_b_bias,
    output wire [`NPU_RSHIFT_BUS_W-1:0] o_b_rshift,
    output wire                         o_b_rshift_valid,
    input  wire [`NPU_GROUP_W-1:0]      i_b_result_group_idx,

    // ---- shared B-core result / issue feedback ----
    input  wire                       i_b_out_valid,
    output wire                       o_b_out_ready,
    input  wire [`NPU_OUT_BUS_W-1:0] i_b_out_data,
    input  wire                       i_b_busy,
    input  wire                       i_b_done,
    input  wire [`NPU_CFG_C_W-1:0]   i_b_conv_cin_idx,
    input  wire [`NPU_CFG_C_W-1:0]   i_b_conv_oc_group_idx,
    input  wire [`NPU_CFG_T_W-1:0]   i_b_conv_time_idx,

    // Unified Conv1..9 parameter system. Ignored in standalone mode.
    input  wire i_shared_params_ready,
    input  wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] i_shared_weight,
    input  wire [`NPU_BIAS_BUS_W-1:0]   i_shared_bias,
    input  wire [`NPU_RSHIFT_BUS_W-1:0] i_shared_rshift,
    input  wire i_shared_weight_valid,
    input  wire i_shared_bias_valid,
    input  wire i_shared_rshift_valid,

    // Producer-side parameter read address.  Unlike B's consumer counters,
    // these indices advance when the local operand register captures a
    // response, allowing the next synchronous BRAM read to overlap the
    // current operand's trip through the two-entry frontend/common pipeline.
    output wire [`NPU_CFG_C_W-1:0] o_param_issue_cin_idx,
    output wire [`NPU_GROUP_W-1:0] o_param_issue_group_idx,

    // ---- debug/scoreboard taps (xsim only, no functional role) ----
    output wire                       dbg_step_fire,
    output wire [`NPU_CFG_T_W-1:0]    dbg_time_idx,
    output wire [`NPU_CFG_C_W-1:0]    dbg_oc_group_idx,
    output wire [`NPU_CFG_C_W-1:0]    dbg_cin_idx,
    output wire signed [15:0]         dbg_x0,
    output wire signed [15:0]         dbg_x1,
    output wire signed [15:0]         dbg_x2,
    // Bank-redesign-only addition (not in the pre-redesign adapter): exposes
    // the weight-bank readout so a TB can scoreboard it through the actual
    // functional read path (dbg_step_fire already implies weight_rd_valid
    // was 1 that cycle, since step_fire=act_row_valid&&act_ready_w and
    // act_row_valid=act_row_valid_core&&weight_rd_valid) instead of peeking
    // a since-removed internal `weight_bank` array.
    output wire [`NPU_WEIGHT_BUS_W-1:0] dbg_sel_weight
);

    localparam [7:0] ERR_LOAD_TKEEP   = 8'h10;
    localparam [7:0] ERR_LOAD_TLAST   = 8'h11;
    localparam [7:0] ERR_LOAD_LENGTH  = 8'h12;
    localparam [7:0] ERR_RUN_TKEEP    = 8'h13;
    localparam [7:0] ERR_RUN_TLAST    = 8'h14;
    localparam [7:0] ERR_RUN_LENGTH   = 8'h15;
    localparam [7:0] ERR_CFG_INVALID  = 8'h16;

    localparam WBANK_STRIDE = 128; // fixed regardless of i_cin, design doc S1.1.1

    // weight_bank depth scales with MAX_OC_GROUP (WBANK_STRIDE itself does
    // not, per the comment above). Address construction below
    // ({group[3:0], cin[6:0]}, 11-bit) is unchanged - cfg_valid already
    // enforces i_num_oc_group <= MAX_OC_GROUP, so the group sub-field's
    // upper bits are always zero at runtime and the resulting address
    // never exceeds WBANK_DEPTH-1. This depth used to be hardcoded to 2048
    // (the full MAX_OC_GROUP=16 case) regardless of the actual MAX_OC_GROUP
    // parameter, which made weight_bank a fixed 16x2048x48=1,572,864-bit
    // array irrespective of this IP's real config - that silently exceeded
    // Vivado's 1,000,000-bit synthesis frontend variable-size limit
    // (caught at synth_design time, not by xsim - xsim has no such limit).
    // At MAX_OC_GROUP=4 (the real Conv2 descriptor), WBANK_DEPTH=512 gives
    // 16x512x48=393,216 bits, safely under the limit.
    localparam WBANK_DEPTH = MAX_OC_GROUP * WBANK_STRIDE;

    // ============================================================
    // SOFT_RESET / error -> registered one-cycle-low pulse on B's rst_n
    // (identical to npu_conv1_backend_adapter.v; malformed-frame reset is
    // mandatory or B stays busy forever and every later op hangs).
    // ============================================================
    reg b_rst_n_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            b_rst_n_reg <= 1'b0;
        else if (i_soft_reset_pulse || o_error_pulse)
            b_rst_n_reg <= 1'b0;
        else
            b_rst_n_reg <= 1'b1;
    end
    wire adapter_clear = !rst_n || !b_rst_n_reg;
    assign o_b_reset_request = i_soft_reset_pulse || o_error_pulse;

    // ============================================================
    // START acceptance / config validity (design doc S4.2 pattern, applied
    // at adapter scope instead of CSR scope).
    // ============================================================
    reg op_active;

    // Scope note (reviewer P2): the RUN-phase fill logic below pushes both
    // halfwords of every LOAD-phase beat unconditionally (no partial-beat
    // path for activation), and its expected-beat-count math assumes
    // Tconv*Cin is even. Real Conv2-9 Cin is always 64 or 128 (even), so
    // this is not a limitation for the intended target - but a generic
    // adapter contract must not silently accept an odd Cin it cannot
    // actually stream correctly. Reject it explicitly at START instead.
    // i_conv_out_len==0 is also rejected here - the last-index/expected-
    // beat arithmetic below (e.g. conv_out_len_r-1) would underflow.
    wire cfg_valid = (i_num_oc_group != 5'd0) && (i_num_oc_group <= MAX_OC_GROUP[4:0]) &&
                      (i_cin != {`NPU_CFG_C_W{1'b0}}) && (i_cin <= MAX_CIN[`NPU_CFG_C_W-1:0]) &&
                      (i_cin[1:0] == 2'b00) &&
                      (i_conv_out_len != {`NPU_CFG_T_W{1'b0}});

    wire start_accept = i_start_pulse && !op_active && !adapter_clear && cfg_valid;
    wire start_reject = i_start_pulse && !op_active && !adapter_clear && !cfg_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            op_active <= 1'b0;
        else if (adapter_clear)
            op_active <= 1'b0;
        else if (start_accept)
            op_active <= 1'b1;
        else if (o_done || o_error_pulse)
            op_active <= 1'b0;
    end

    assign o_busy = op_active;

    // Latched descriptor bounds - stable for the whole operation.
    reg [`NPU_CFG_C_W-1:0] cin_r;
    // Predecode the inclusive channel bound at START.  The weight loader
    // advances two canonical elements per beat; keeping the subtractor off
    // both cascaded next-index functions removes a repeated CARRY cone from
    // the live BRAM-address path.
    reg [`NPU_CFG_C_W-1:0] cin_last_r;
    reg [4:0]              num_oc_group_r;
    reg [`NPU_CFG_T_W-1:0] conv_out_len_r;
    // Inclusive final convolution time index, predecoded once at START so
    // the live operand issue path contains only an equality comparator.
    reg [`NPU_CFG_T_W-1:0] conv_last_t_r;
    reg                    pool_enable_r;
    // out_len_r / last_output_group_r used to be computed here (a `/3`
    // division) and then in CFG_DERIVE (a multiply) respectively - both were
    // found to independently violate OOC timing (Docs/C_to_B_ConvOutLen_
    // ConfigTiming_Review_Request_v1.md, and the follow-up OOC re-measurement
    // that showed relocating the division into CFG_DERIVE just moved the
    // same violation onto out_len_r_reg instead of removing it - a division/
    // multiply on this bit width simply does not fit in one 10ns cycle
    // regardless of which register captures the result). Per the agreed
    // follow-up decision, both are now manifest-authoritative values supplied
    // directly by the caller (i_conv_post_out_len/i_last_output_group) and
    // are pure latches here, same as cin_r/num_oc_group_r - no arithmetic of
    // any kind on the adapter's or B's START/configuration path any more.
    reg [`NPU_CFG_T_W-1:0] out_len_r;
    reg [31:0]              last_output_group_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cin_r               <= {`NPU_CFG_C_W{1'b0}};
            cin_last_r          <= {`NPU_CFG_C_W{1'b0}};
            num_oc_group_r      <= 5'd0;
            conv_out_len_r      <= {`NPU_CFG_T_W{1'b0}};
            conv_last_t_r       <= {`NPU_CFG_T_W{1'b0}};
            pool_enable_r       <= 1'b0;
            out_len_r           <= {`NPU_CFG_T_W{1'b0}};
            last_output_group_r <= 32'd0;
        end
        else if (start_accept) begin
            cin_r               <= i_cin;
            cin_last_r          <= i_cin - {{(`NPU_CFG_C_W-1){1'b0}}, 1'b1};
            num_oc_group_r      <= i_num_oc_group;
            conv_out_len_r      <= i_conv_out_len;
            conv_last_t_r       <= i_conv_out_len - {{(`NPU_CFG_T_W-1){1'b0}}, 1'b1};
            pool_enable_r       <= i_pool_enable;
            out_len_r           <= i_conv_post_out_len;
            last_output_group_r <= i_last_output_group;
        end
    end

    wire [`NPU_CFG_C_W-1:0] cout_r = {num_oc_group_r, 4'b0000}; // = num_oc_group_r * 16

    // ============================================================
    // LOAD phase FSM: weight -> bias -> rshift, strictly sequential
    // (structure mirrors npu_conv1_backend_adapter.v exactly; per-group beat
    // counts are now derived from cin_r/num_oc_group_r instead of fixed
    // localparams).
    // ============================================================
    localparam [2:0] PHASE_LOAD_WEIGHT   = 3'd0;
    localparam [2:0] PHASE_LOAD_BIAS     = 3'd1;
    localparam [2:0] PHASE_LOAD_RSHIFT   = 3'd2;
    localparam [2:0] PHASE_PREFETCH_REQ  = 3'd3;
    localparam [2:0] PHASE_PREFETCH_LATCH = 3'd4;
    localparam [2:0] PHASE_RUN           = 3'd5;
    // The final rshift beat is captured into a registered BRAM-write
    // command.  Give that command one cycle to drain before issuing the
    // group-0 synchronous prefetch, including the MAX_OC_GROUP=1 case where
    // the write and read addresses are identical.
    localparam [2:0] PHASE_RSHIFT_DRAIN  = 3'd6;

    reg [2:0]  phase;
    reg [31:0] beat_in_group;
    reg [4:0]  group_idx;      // 0..num_oc_group_r-1
    reg [31:0] load_beat_cnt;

    wire [31:0] bias_beats_per_group   = 32'd32; // 16 lanes * 2 beats/lane, cin-independent
    wire [31:0] rshift_beats_per_group = 32'd4;  // 16B/group / 4B, cin-independent

    // ============================================================
    // CFG_DERIVE: 2-stage pipelined descriptor-derived register bank
    // (Docs/C_WeightBank_16Way_BRAM_Redesign_Contract_v1.md addendum -
    // OOC post-synth timing found WNS=-6.146ns on cin_r -> 24*cin_r ->
    // *num_oc_group_r -> expected_last_beat -> 48 memory ENARDEN, a
    // 2-chained-32-bit-multiply path whose LOGIC delay alone (10.083ns)
    // already exceeds the 10ns/100MHz budget - not a fanout artifact, so a
    // single derive cycle would still leave one violating path (into
    // whichever register captures the result). Split into two stages so
    // neither ever chains two multiplies in one cycle:
    //   stage 1 (op_active's 1st cycle, cin_r/num_oc_group_r/conv_out_len_r/
    //     pool_enable_r already latched by start_accept the cycle before):
    //     everything that is only ONE multiply/divide deep from an already-
    //     registered operand.
    //   stage 2 (next cycle): weight_total_beats_r needs a second op that
    //     reads a stage-1 result (weight_beats_per_group_r), so it is only
    //     one op deep from an already-registered stage-1 operand.
    // out_len_r/last_output_group_r are NOT part of this pipeline any more -
    // they are manifest-authoritative descriptor inputs latched directly at
    // start_accept above (see i_conv_post_out_len/i_last_output_group), no
    // arithmetic left to stage.
    // cfg_ready gates s_axis_tready (below) so LOAD cannot start, and
    // nothing reads these registers, until stage 2 completes.
    // ============================================================
    reg        cfg_stage1_done;
    reg        cfg_ready;
    // ConvN bounds are fixed by the unified descriptor envelope:
    //   24 * MAX_CIN(128)              = 3,072  (12 bits)
    //   3,072 * MAX_OC_GROUP(16)       = 49,152 (16 bits)
    // Keeping these values at 32 bits made Vivado build a two-DSP cascade
    // for the second product even though the upper bits can never be used.
    reg [11:0] weight_beats_per_group_r;
    reg [15:0] weight_total_beats_r;
    reg [31:0] bias_total_beats_r;
    reg [31:0] rshift_total_beats_r;
    reg [31:0] run_expected_beats_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_stage1_done           <= 1'b0;
            cfg_ready                 <= 1'b0;
            weight_beats_per_group_r  <= 12'd0;
            weight_total_beats_r      <= 16'd0;
            bias_total_beats_r        <= 32'd0;
            rshift_total_beats_r      <= 32'd0;
            run_expected_beats_r      <= 32'd0;
        end
        else if (adapter_clear || !op_active) begin
            cfg_stage1_done           <= 1'b0;
            cfg_ready                 <= 1'b0;
            weight_beats_per_group_r  <= 12'd0;
            weight_total_beats_r      <= 16'd0;
            bias_total_beats_r        <= 32'd0;
            rshift_total_beats_r      <= 32'd0;
            run_expected_beats_r      <= 32'd0;
        end
        else if (!cfg_stage1_done) begin
            weight_beats_per_group_r <= 12'd24 * cin_r; // 16*3*2/4 = 24 per cin
            bias_total_beats_r       <= bias_beats_per_group * {27'b0, num_oc_group_r};
            rshift_total_beats_r     <= rshift_beats_per_group * {27'b0, num_oc_group_r};
            // Manifest/descriptor predecode.  The old expression implemented
            // Tconv*Cin/2 as a runtime multiply and consumed the only DSP
            // outside the 48-lane MAC array in the unified top.
            run_expected_beats_r     <= i_activation_beats;
            cfg_stage1_done          <= 1'b1;
        end
        else if (!cfg_ready) begin
            weight_total_beats_r <= weight_beats_per_group_r * num_oc_group_r;
            cfg_ready            <= 1'b1;
        end
    end

    wire is_load_phase = !USE_SHARED_PARAMS && ((phase == PHASE_LOAD_WEIGHT) ||
                         (phase == PHASE_LOAD_BIAS) ||
                         (phase == PHASE_LOAD_RSHIFT));

    wire [31:0] beats_per_group_m1 = (phase == PHASE_LOAD_WEIGHT) ? ({20'd0, weight_beats_per_group_r} - 32'd1) :
                                      (phase == PHASE_LOAD_BIAS)   ? (bias_beats_per_group - 32'd1)   :
                                                                      (rshift_beats_per_group - 32'd1);
    wire [31:0] total_beats_m1     = (phase == PHASE_LOAD_WEIGHT) ? ({16'd0, weight_total_beats_r} - 32'd1) :
                                      (phase == PHASE_LOAD_BIAS)   ? (bias_total_beats_r - 32'd1)   :
                                                                      (rshift_total_beats_r - 32'd1);

    wire load_fire          = s_axis_tvalid && s_axis_tready && is_load_phase;
    wire expected_last_beat = (load_beat_cnt == total_beats_m1);

    wire load_tkeep_bad     = load_fire && (s_axis_tkeep != 4'hF);
    wire load_tlast_early   = load_fire && s_axis_tlast && !expected_last_beat;
    wire load_tlast_missing = load_fire && expected_last_beat && !s_axis_tlast;
    wire load_error         = load_tkeep_bad || load_tlast_early || load_tlast_missing;

    wire [7:0] load_error_code = load_tkeep_bad     ? ERR_LOAD_TKEEP  :
                                  load_tlast_early   ? ERR_LOAD_TLAST  :
                                  load_tlast_missing ? ERR_LOAD_LENGTH :
                                                        8'h00;

    reg b_start_pulse_reg;

    // weight bank storage (48 lane x tap XPM instances) is declared further
    // down, right before its first use point (weight_rd_addr) - xvlog's
    // SystemVerilog analysis enforces declare-before-use even at module
    // scope, and the write-side demux needs w_lane0/w_addr0/w_lane1/w_addr1/
    // w_k_idx/w_k_n1 (declared below) plus weight_rd_addr (declared with the
    // read side) as inputs. See "weight bank: lane x tap ..." below.

    // Bias/rshift payloads are stored in lane/word-local synchronous BRAM
    // banks.  They are prefetched into small result/issue-aligned caches at
    // group boundaries; no wide group-indexed FF mux remains in RUN.
    localparam integer PARAM_GROUP_AW = (MAX_OC_GROUP <= 2) ? 1 : $clog2(MAX_OC_GROUP);
    wire [47:0] bias_bank_rd [0:15];
    wire [31:0] rshift_bank_rd [0:3];

    // Canonical [OC][IC][K] stream position for the weight phase, K-fastest,
    // advanced by exactly 2 elements (one beat) per accepted LOAD beat -
    // legal because LOAD requires tkeep=4'hF (both halfwords always present).
    reg [1:0]               w_k_idx;
    reg [`NPU_CFG_C_W-1:0]  w_ic_idx;
    reg [`NPU_CFG_C_W-1:0]  w_oc_idx;

    function [2+2*`NPU_CFG_C_W-1:0] w_idx_next;
        input [1:0]              k_in;
        input [`NPU_CFG_C_W-1:0] ic_in;
        input [`NPU_CFG_C_W-1:0] oc_in;
        input [`NPU_CFG_C_W-1:0] cin_last_in;
        reg [1:0]              k_o;
        reg [`NPU_CFG_C_W-1:0] ic_o, oc_o;
        begin
            if (k_in == 2'd2) begin
                k_o = 2'd0;
                if (ic_in == cin_last_in) begin
                    ic_o = {`NPU_CFG_C_W{1'b0}};
                    oc_o = oc_in + 1'b1;
                end
                else begin
                    ic_o = ic_in + 1'b1;
                    oc_o = oc_in;
                end
            end
            else begin
                k_o  = k_in + 2'd1;
                ic_o = ic_in;
                oc_o = oc_in;
            end
            w_idx_next = {k_o, ic_o, oc_o};
        end
    endfunction

    wire [2+2*`NPU_CFG_C_W-1:0] w_step1 = w_idx_next(w_k_idx, w_ic_idx, w_oc_idx, cin_last_r);
    wire [1:0]              w_k_n1  = w_step1[2+2*`NPU_CFG_C_W-1 -: 2];
    wire [`NPU_CFG_C_W-1:0] w_ic_n1 = w_step1[2*`NPU_CFG_C_W-1 -: `NPU_CFG_C_W];
    wire [`NPU_CFG_C_W-1:0] w_oc_n1 = w_step1[`NPU_CFG_C_W-1:0];

    wire [2+2*`NPU_CFG_C_W-1:0] w_step2 = w_idx_next(w_k_n1, w_ic_n1, w_oc_n1, cin_last_r);
    wire [1:0]              w_k_n2  = w_step2[2+2*`NPU_CFG_C_W-1 -: 2];
    wire [`NPU_CFG_C_W-1:0] w_ic_n2 = w_step2[2*`NPU_CFG_C_W-1 -: `NPU_CFG_C_W];
    wire [`NPU_CFG_C_W-1:0] w_oc_n2 = w_step2[`NPU_CFG_C_W-1:0];

    wire [3:0] w_lane0  = w_oc_idx[3:0];
    wire [3:0] w_group0 = w_oc_idx[`NPU_CFG_C_W-1:4];
    wire [10:0] w_addr0 = {w_group0, w_ic_idx[6:0]};

    wire [3:0] w_lane1  = w_oc_n1[3:0];
    wire [3:0] w_group1 = w_oc_n1[`NPU_CFG_C_W-1:4];
    wire [10:0] w_addr1 = {w_group1, w_ic_n1[6:0]};

    integer bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase             <= PHASE_LOAD_WEIGHT;
            beat_in_group     <= 32'd0;
            group_idx         <= 5'd0;
            load_beat_cnt     <= 32'd0;
            b_start_pulse_reg <= 1'b0;
            w_k_idx           <= 2'd0;
            w_ic_idx          <= {`NPU_CFG_C_W{1'b0}};
            w_oc_idx          <= {`NPU_CFG_C_W{1'b0}};
        end
        else if (adapter_clear || !op_active) begin
            phase             <= PHASE_LOAD_WEIGHT;
            beat_in_group     <= 32'd0;
            group_idx         <= 5'd0;
            load_beat_cnt     <= 32'd0;
            b_start_pulse_reg <= 1'b0;
            w_k_idx           <= 2'd0;
            w_ic_idx          <= {`NPU_CFG_C_W{1'b0}};
            w_oc_idx          <= {`NPU_CFG_C_W{1'b0}};
        end
        else begin
            b_start_pulse_reg <= 1'b0;

            if (USE_SHARED_PARAMS && i_shared_params_ready) begin
                phase             <= PHASE_RUN;
                b_start_pulse_reg <= 1'b1;
            end
            else if (load_fire && !load_error) begin
                case (phase)
                    PHASE_LOAD_WEIGHT: begin
                        // Storage write is issued through the registered
                        // weight-commit command below.  This branch advances
                        // only the accepted-beat (k,ic,oc) index state.
                        w_k_idx  <= w_k_n2;
                        w_ic_idx <= w_ic_n2;
                        w_oc_idx <= w_oc_n2;
                    end
                    // PHASE_LOAD_BIAS/PHASE_LOAD_RSHIFT no longer write here -
                    // the actual storage write is driven by the registered
                    // one-hot group/lane state (declared below, right after
                    // this FSM) instead of beat_in_group/group_idx compared
                    // live every cycle (Docs/C_BiasRshift_LoadStorage_
                    // Redesign_Contract_v1.md S6). beat_in_group/group_idx
                    // themselves are unchanged and still drive frame timing/
                    // TLAST/phase transitions (contract S4) - only the
                    // storage-write decode moved off of them.
                    default: ;
                endcase

                load_beat_cnt <= load_beat_cnt + 32'd1;

                if (beat_in_group == beats_per_group_m1) begin
                    beat_in_group <= 32'd0;
                    group_idx     <= group_idx + 5'd1;
                end
                else begin
                    beat_in_group <= beat_in_group + 32'd1;
                end

                if (expected_last_beat) begin
                    beat_in_group <= 32'd0;
                    group_idx     <= 5'd0;
                    load_beat_cnt <= 32'd0;
                    // weight-phase element counters intentionally NOT reset
                    // here on the weight->bias transition - they are only
                    // meaningful during PHASE_LOAD_WEIGHT and are reset by
                    // the op-start / adapter_clear branch above for the
                    // next operation.
                    case (phase)
                        PHASE_LOAD_WEIGHT: phase <= PHASE_LOAD_BIAS;
                        PHASE_LOAD_BIAS:   phase <= PHASE_LOAD_RSHIFT;
                        PHASE_LOAD_RSHIFT: begin
                            phase <= PHASE_RSHIFT_DRAIN;
                        end
                        default: ;
                    endcase
                end
            end

            // Synchronous BRAM reads are explicit non-AXIS phases.  Both
            // group-0 caches are ready before B receives START.
            if (phase == PHASE_RSHIFT_DRAIN)
                phase <= PHASE_PREFETCH_REQ;
            else if (phase == PHASE_PREFETCH_REQ)
                phase <= PHASE_PREFETCH_LATCH;
            else if (phase == PHASE_PREFETCH_LATCH) begin
                phase             <= PHASE_RUN;
                b_start_pulse_reg <= 1'b1;
            end
        end
    end

    // ============================================================
    // bias/rshift LOAD storage write (Docs/C_BiasRshift_LoadStorage_
    // Redesign_Contract_v1.md). Registered one-hot group/lane selection,
    // shifted incrementally in lockstep with beat_in_group/group_idx above
    // (never recomputed via comparison from them) - S6 observable contract:
    // exactly one storage target selected per accepted beat, selection built
    // from registered state, no variable packed part-select on the write.
    //
    // bias: even beat stages 32 LSBs; odd beat commits {odd_beat[15:0],
    // staged_32b} into bias_lane[group][lane] in one write (contract S2 byte
    // order/padding-ignore unchanged - bits[31:16] of the odd beat are never
    // read). lane advances only on the commit (odd) beat.
    // rshift: each beat commits one 32-bit word (4 lanes) into
    // rshift_word[group][word] (contract S3, unchanged).
    // group advances on the shared group-boundary condition (beat_in_group
    // == beats_per_group_m1), mirroring group_idx above exactly, for ALL
    // phases uniformly (harmless/unused during WEIGHT, same as group_idx
    // itself). Re-armed to group0/lane0/word0 on every phase transition
    // (expected_last_beat) - contract S6b: only this small control state is
    // reset; bias_lane/rshift_word data itself is never reset (contract S6b/
    // §0 - avoids adding async clear to the new storage, which would worsen
    // the separate async_default/b_rst_n_reg fan-out issue).
    // ============================================================
    reg [15:0]             ldw_bias_lane_onehot;
    reg [3:0]              ldw_rshift_word_onehot;
    reg                    ldw_bias_half;   // 0 = staging beat, 1 = commit beat
    reg [31:0]             bias_staging;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ldw_bias_lane_onehot   <= 16'h0001;
            ldw_rshift_word_onehot <= 4'h1;
            ldw_bias_half          <= 1'b0;
        end
        else if (adapter_clear || !op_active) begin
            ldw_bias_lane_onehot   <= 16'h0001;
            ldw_rshift_word_onehot <= 4'h1;
            ldw_bias_half          <= 1'b0;
        end
        else if (load_fire && !load_error) begin
            case (phase)
                PHASE_LOAD_BIAS: begin
                    if (!ldw_bias_half) begin
                        bias_staging  <= s_axis_tdata;
                        ldw_bias_half <= 1'b1;
                    end
                    else begin
                        ldw_bias_half <= 1'b0;
                        if (beat_in_group == beats_per_group_m1)
                            ldw_bias_lane_onehot <= 16'h0001;
                        else
                            ldw_bias_lane_onehot <= {ldw_bias_lane_onehot[14:0], ldw_bias_lane_onehot[15]};
                    end
                end
                PHASE_LOAD_RSHIFT: begin
                    if (beat_in_group == beats_per_group_m1)
                        ldw_rshift_word_onehot <= 4'h1;
                    else
                        ldw_rshift_word_onehot <= {ldw_rshift_word_onehot[2:0], ldw_rshift_word_onehot[3]};
                end
                default: ;
            endcase

            if (expected_last_beat) begin
                ldw_bias_lane_onehot   <= 16'h0001;
                ldw_rshift_word_onehot <= 4'h1;
                ldw_bias_half          <= 1'b0;
            end
        end
    end

    // The BRAM instances are declared beside their RUN-side prefetch/cache
    // consumers after B's issue/result group indices are declared.

    // ============================================================
    // RUN phase: 4-row line buffer.  Three rows form the active K=3 window
    // while the fourth row is filled ahead of the next time step, hiding the
    // AXI activation refill latency behind the current Cin/group sweep.
    // The admission path retains the existing 1-beat
    // admission pipeline (Docs/C_Admission_ElasticBuffer_Redesign_Contract_
    // v1.md): accept-time validation (Cycle N, against RESERVE state) is
    // registered as buf_valid/admit_ok and consumed one beat later by the
    // commit (Cycle N+1), so line_buf's CE is gated only by a registered
    // AND of two 1-bit regs - not by a live 32-bit compare - breaking the
    // run_expected_last_beat -> line_buf CE same-cycle chain that produced
    // the Conv2 P&R violation (contract S0, WNS=-0.700ns). Reservation
    // counters (reserve_*) and physical counters (fill_row_ptr/fill_ch_idx/
    // rows_filled/run_beat_cnt) use the IDENTICAL self-referential
    // transition function, offset by exactly one admitted beat (buffer
    // depth is exactly 1 - contract S2/S5), so physical always addresses
    // the beat currently staged in the buffer entry without needing its own
    // latched address.
    // ============================================================
    // S11 bank-local storage (see GEN_BANKROW/GEN_BANKCOL below): line_buf
    // is no longer one shared array. A first attempt kept it shared and
    // wrote it from 24 separate (row,bank) always blocks using a computed
    // {bank_const, local_dynamic} address - Vivado's elaborator could not
    // prove those 24 write-address ranges are disjoint (DRC MDRV-1,
    // "multiple drivers", on every line_buf bit), even though they
    // mathematically never overlap. Fixed by giving each (row,bank) its
    // own PHYSICALLY SEPARATE 16-entry register array (declared inside its
    // own generate-block scope, below) - textually impossible for the tool
    // to confuse with another instance's array. The read side is rebuilt
    // as an explicit two-level combinational mux (bank_rdata_flat) instead
    // of a single shared-array read.

    reg [1:0]               fill_row_ptr;   // physical (commit-time)
    reg [`NPU_CFG_C_W-1:0]  fill_ch_idx;    // physical (commit-time), 0..cin_r step 2
    reg [`NPU_CFG_T_W-1:0]  rows_filled;    // physical (commit-time) - unchanged consumer contract (S8: act_row_valid_core)
    reg [31:0]              run_beat_cnt;   // physical (commit-time)

    reg [1:0]               reserve_fill_row_ptr;   // reservation (accept-time)
    reg [`NPU_CFG_C_W-1:0]  reserve_fill_ch_idx;    // reservation (accept-time)
    reg [`NPU_CFG_T_W-1:0]  reserve_rows_filled;    // reservation (accept-time)
    reg [31:0]              reserve_run_beat_cnt;   // reservation (accept-time)

    reg [1:0]               read_row_ptr;
    reg [`NPU_CFG_T_W-1:0]  compute_t;     // current window's time index
    reg [`NPU_CFG_T_W-1:0]  rows_consumed; // count of fully-consumed windows
    reg [`NPU_CFG_C_W-1:0]  issue_cin_idx_r;
    reg [`NPU_GROUP_W-1:0]  issue_group_idx_r;
    reg                     capture_arm;
    reg                     producer_done_r;

    wire run_axis_fire = s_axis_tvalid && s_axis_tready && (phase == PHASE_RUN);

    // Reviewer P0 fix: this must NOT be NPU_CFG_T_W (20 bits). Real Conv2
    // (Tconv=106667, Cin=64) needs Tconv*Cin*2/4 = 3,413,344 beats, which
    // needs 22 bits - a 20-bit run_expected_beats/run_beat_cnt silently
    // truncates that to 267,616 (3,413,344 mod 2^20), turning every
    // well-formed full-size Conv2 activation frame into a spurious early-
    // TLAST/length error. Widened to 32 bits (operands zero-extended to 32
    // bits before the multiply so the product itself isn't truncated
    // either - max product is ~27M for MAX_CIN=128, well within 32 bits).
    // (Tconv*Cin is always even for the Cin values this module supports -
    // see header note; run_expected_beats_r is therefore exact, no partial
    // final beat in the well-formed case. The malformed-frame regression
    // constructs a deliberately wrong length to exercise the error path.)
    // Computation itself now lives in the CFG_DERIVE pipeline above
    // (run_expected_beats_r) instead of a live combinational multiply - see
    // that block's comment for why.
    //
    // accept-time (Cycle N) admission check, against RESERVE state (contract
    // S2/S6) - this is the successor of the old run_expected_last_beat/
    // run_tkeep_bad/run_tlast_early/run_tlast_missing chain, now indexed by
    // reserve_run_beat_cnt instead of the physical (commit-time) counter.
    wire accept_expected_last = (reserve_run_beat_cnt == run_expected_beats_r - 1'b1);

    wire accept_tkeep_bad     = run_axis_fire && (s_axis_tkeep != 4'hF) && !(s_axis_tlast && s_axis_tkeep == 4'h3);
    wire accept_tlast_early   = run_axis_fire && s_axis_tlast && !accept_expected_last;
    wire accept_tlast_missing = run_axis_fire && accept_expected_last && !s_axis_tlast;
    wire run_error            = accept_tkeep_bad || accept_tlast_early || accept_tlast_missing;

    wire [7:0] run_error_code = accept_tkeep_bad     ? ERR_RUN_TKEEP  :
                                 accept_tlast_early   ? ERR_RUN_TLAST  :
                                 accept_tlast_missing ? ERR_RUN_LENGTH :
                                                         8'h00;

    // o_error_pulse/o_error_code fire at accept time, same cycle as before
    // (contract S6/S7) - o_error_pulse feeds b_rst_n_reg.D (line ~167), a
    // separate setup-timing path from the commit/CE path (contract S6
    // correction: NOT timing-free, but a distinct and much narrower fan-out
    // path - verified separately in OOC/P&R, not addressed by this change).
    assign o_error_pulse = start_reject ||
                           (op_active && ((USE_SHARED_PARAMS ? 1'b0 : load_error) || run_error));
    assign o_error_code  = start_reject   ? ERR_CFG_INVALID :
                            is_load_phase ? load_error_code :
                                            run_error_code;

    wire accept_admit_ok = run_axis_fire && !run_error;   // combinational, Cycle N only

    // row-complete decision, computed independently at accept time (drives
    // reservation advance) and at commit time (drives physical advance) -
    // identical transition function applied to the two counters (contract
    // S5).
    wire reserve_row_complete  = (reserve_fill_ch_idx + {{(`NPU_CFG_C_W-2){1'b0}}, 2'd2} >= cin_r);
    wire physical_row_complete = (fill_ch_idx        + {{(`NPU_CFG_C_W-2){1'b0}}, 2'd2} >= cin_r);

    // elastic buffer entry: registered at accept (N), consumed at commit
    // (N+1) - contract S2/S4/S6.
    reg        buf_valid;
    reg        admit_ok;

    // The AXIS beat already contains one adjacent channel pair.  Keep that
    // natural 32-bit word intact and write it into one of four row BRAMs at
    // address channel/2.  The fourth memory is the prefetch row that hides
    // the time-step refill gap.
    localparam LINE_WRITE_DEPTH = MAX_CIN / 2;
    localparam LINE_READ_DEPTH  = MAX_CIN / `NPU_PIN;
    localparam LINE_WRITE_ADDR_W = $clog2(LINE_WRITE_DEPTH);
    localparam LINE_READ_ADDR_W  = $clog2(LINE_READ_DEPTH);
    reg [31:0] buf_tdata;

    // Backpressure (contract S3): reservation room uses RESERVE rows_filled
    // (one beat ahead of physical) so an admitted-but-not-yet-committed beat
    // already counts as occupying its row slot.  At most the active three-row
    // window is reserved; the fourth physical row remains the fill-ahead slot.
    wire run_room       = (reserve_rows_filled < conv_out_len_r) &&
                           ((reserve_rows_filled - rows_consumed) < {{(`NPU_CFG_T_W-2){1'b0}}, 2'd3});
    // Enqueue room (contract S3): buffer is free, OR the resident entry
    // commits on this very edge - lets accept and commit overlap on the
    // same cycle so the normal path keeps 1 beat/clock throughput.
    wire run_enqueue_ok = !buf_valid || (buf_valid && admit_ok);
    wire run_hw_tready  = run_room && run_enqueue_ok;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reserve_fill_row_ptr <= 2'd0;
            reserve_fill_ch_idx  <= {`NPU_CFG_C_W{1'b0}};
            reserve_rows_filled  <= {`NPU_CFG_T_W{1'b0}};
            reserve_run_beat_cnt <= 32'd0;
            buf_valid    <= 1'b0;
            admit_ok     <= 1'b0;
            fill_row_ptr <= 2'd0;
            fill_ch_idx  <= {`NPU_CFG_C_W{1'b0}};
            rows_filled  <= {`NPU_CFG_T_W{1'b0}};
            run_beat_cnt <= 32'd0;
        end
        else if (adapter_clear || (phase != PHASE_RUN)) begin
            reserve_fill_row_ptr <= 2'd0;
            reserve_fill_ch_idx  <= {`NPU_CFG_C_W{1'b0}};
            reserve_rows_filled  <= {`NPU_CFG_T_W{1'b0}};
            reserve_run_beat_cnt <= 32'd0;
            buf_valid    <= 1'b0;
            admit_ok     <= 1'b0;
            fill_row_ptr <= 2'd0;
            fill_ch_idx  <= {`NPU_CFG_C_W{1'b0}};
            rows_filled  <= {`NPU_CFG_T_W{1'b0}};
            run_beat_cnt <= 32'd0;
        end
        else begin
            // ---- Cycle N: accept-time admission - register the buffer entry ----
            buf_valid <= run_axis_fire;
            admit_ok  <= accept_admit_ok;
            if (accept_admit_ok)
                buf_tdata <= s_axis_tdata;

            if (accept_admit_ok) begin
                reserve_run_beat_cnt <= reserve_run_beat_cnt + 1'b1;
                if (reserve_row_complete) begin
                    reserve_fill_ch_idx  <= {`NPU_CFG_C_W{1'b0}};
                    reserve_fill_row_ptr <= (reserve_fill_row_ptr == 2'd3) ? 2'd0 : reserve_fill_row_ptr + 2'd1;
                    reserve_rows_filled  <= reserve_rows_filled + 1'b1;
                end
                else begin
                    reserve_fill_ch_idx <= reserve_fill_ch_idx + {{(`NPU_CFG_C_W-2){1'b0}}, 2'd2};
                end
            end

            // ---- Cycle N+1: commit.  The row-BRAM write enable/address are
            // derived from these unchanged physical counters. ----
            if (buf_valid && admit_ok) begin
                run_beat_cnt <= run_beat_cnt + 1'b1;

                if (physical_row_complete) begin
                    fill_ch_idx  <= {`NPU_CFG_C_W{1'b0}};
                    fill_row_ptr <= (fill_row_ptr == 2'd3) ? 2'd0 : fill_row_ptr + 2'd1;
                    rows_filled  <= rows_filled + 1'b1;
                end
                else begin
                    fill_ch_idx <= fill_ch_idx + {{(`NPU_CFG_C_W-2){1'b0}}, 2'd2};
                end
            end
        end
    end

    // Four row memories provide the three simultaneous tap reads plus one
    // look-ahead fill slot.  Each memory has one synchronous read port and
    // one write port, so prefetching the free row overlaps current compute.
    // The producer cursor names the response captured on this edge when
    // capture_arm is set.  In shared-parameter mode the BRAM address ports
    // simultaneously receive the following cursor, creating a true
    // one-request-per-clock synchronous-read pipeline.
    wire current_last_cin = ((issue_cin_idx_r + 3'd4) >= cin_r);
    wire current_last_group = (issue_group_idx_r == num_oc_group_r - 1'b1);
    wire current_last_window_step = current_last_cin && current_last_group;
    wire current_final_step = current_last_window_step && (compute_t == conv_last_t_r);

    wire [`NPU_CFG_C_W-1:0] next_cin_idx_w = current_last_cin ?
        {`NPU_CFG_C_W{1'b0}} : issue_cin_idx_r + 3'd4;
    wire [`NPU_GROUP_W-1:0] next_group_idx_w = current_last_cin ?
        (current_last_group ? {`NPU_GROUP_W{1'b0}} : issue_group_idx_r + 1'b1) :
        issue_group_idx_r;
    wire [`NPU_CFG_T_W-1:0] next_compute_t_w = current_last_window_step ?
        compute_t + 1'b1 : compute_t;
    wire [1:0] next_read_row_ptr_w = current_last_window_step ?
        ((read_row_ptr == 2'd3) ? 2'd0 : read_row_ptr + 2'd1) : read_row_ptr;

    // Standalone ConvN retains its conservative one-shot cache contract.
    // The packaged unified Top always sets USE_SHARED_PARAMS=1 and therefore
    // uses the pipelined look-ahead cursor below.
    wire request_lookahead = (USE_SHARED_PARAMS != 0) && capture_arm;
    wire [`NPU_CFG_C_W-1:0] request_cin_idx_w = request_lookahead ?
        next_cin_idx_w : issue_cin_idx_r;
    wire [`NPU_GROUP_W-1:0] request_group_idx_w = request_lookahead ?
        next_group_idx_w : issue_group_idx_r;
    wire [`NPU_CFG_T_W-1:0] request_compute_t_w = request_lookahead ?
        next_compute_t_w : compute_t;
    wire [1:0] request_read_row_ptr_w = request_lookahead ?
        next_read_row_ptr_w : read_row_ptr;

    wire [`NPU_CFG_C_W-1:0] cin_idx_w = issue_cin_idx_r;
    wire [LINE_READ_ADDR_W-1:0] line_rd_addr = request_cin_idx_w[LINE_READ_ADDR_W+1:2];
    wire [LINE_WRITE_ADDR_W-1:0] line_wr_addr = fill_ch_idx[LINE_WRITE_ADDR_W:1];
    wire [63:0] line_row_rd [0:3];

    genvar gr;
    generate
        for (gr = 0; gr < 4; gr = gr + 1) begin : GEN_BANKROW
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A(LINE_WRITE_ADDR_W), .ADDR_WIDTH_B(LINE_READ_ADDR_W),
                .AUTO_SLEEP_TIME(0), .BYTE_WRITE_WIDTH_A(32), .CASCADE_HEIGHT(0),
                .CLOCKING_MODE("common_clock"), .ECC_MODE("no_ecc"),
                .MEMORY_INIT_FILE("none"), .MEMORY_INIT_PARAM("0"),
                .MEMORY_OPTIMIZATION("true"), .MEMORY_PRIMITIVE("block"),
                .MEMORY_SIZE(MAX_CIN*16), .MESSAGE_CONTROL(0),
                .READ_DATA_WIDTH_B(64), .READ_LATENCY_B(1),
                .READ_RESET_VALUE_B("0"), .RST_MODE_A("SYNC"), .RST_MODE_B("SYNC"),
                .SIM_ASSERT_CHK(0), .USE_EMBEDDED_CONSTRAINT(0), .USE_MEM_INIT(0),
                .WAKEUP_TIME("disable_sleep"), .WRITE_DATA_WIDTH_A(32),
                .WRITE_MODE_B("no_change")
            ) u_line_row (
                .clka(clk), .ena(1'b1),
                .wea(buf_valid && admit_ok && (fill_row_ptr == gr[1:0])),
                .addra(line_wr_addr), .dina(buf_tdata),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .clkb(clk), .rstb(1'b0), .enb(phase == PHASE_RUN),
                .regceb(1'b1), .addrb(line_rd_addr), .doutb(line_row_rd[gr]),
                .sleep(1'b0), .dbiterrb(), .sbiterrb()
            );
        end
    endgenerate

    reg [LINE_READ_ADDR_W-1:0] line_rd_addr_reg;
    reg line_rd_started;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            line_rd_addr_reg <= {LINE_READ_ADDR_W{1'b0}};
            line_rd_started  <= 1'b0;
        end
        else if (adapter_clear || (phase != PHASE_RUN)) begin
            line_rd_addr_reg <= {LINE_READ_ADDR_W{1'b0}};
            line_rd_started  <= 1'b0;
        end
        else begin
            line_rd_addr_reg <= line_rd_addr;
            line_rd_started  <= 1'b1;
        end
    end
    wire line_rd_valid = line_rd_started && (line_rd_addr == line_rd_addr_reg);

    // ---- producer-side read address.  B's issue counters are consumer
    // state and therefore trail the local/common operand pipeline.  Keeping
    // a small producer cursor here lets the address for operand N+1 reach the
    // synchronous line/weight BRAMs while operand N is being transferred. ----
    wire [`NPU_CFG_C_W-1:0] oc_group_idx_w;
    wire [`NPU_CFG_T_W-1:0] time_idx_w;
    assign oc_group_idx_w = {{(`NPU_CFG_C_W-`NPU_GROUP_W){1'b0}}, issue_group_idx_r};
    assign time_idx_w     = compute_t;
    assign o_param_issue_cin_idx   = request_cin_idx_w;
    assign o_param_issue_group_idx = request_group_idx_w;
    wire                    step_fire;      // driven below

    // The registered producer cursor drives all 48 ConvN weight BRAM read
    // ports. Permit synthesis to replicate the small address cone locally.
    (* max_fanout = 4 *) wire [8:0] weight_rd_addr =
        {request_group_idx_w[3:0], request_cin_idx_w[6:2]};

    // ---- weight bank: lane x tap 16bit synchronous-read banks (Docs/
    // C_WeightBank_16Way_BRAM_Redesign_Contract_v1.md S2/S3 option A,
    // approved 1st RTL structure). Replaces the single 16x2048x48-bit
    // behavioral array + combinational read that made Technology Mapping
    // require tens of GiB for a 48KiB(Conv2)/196KiB(Conv9) real payload
    // (see contract doc S0). 16 lanes x 3 taps = 48 independent 16-bit-wide,
    // WBANK_DEPTH-deep XPM_MEMORY_SDPRAM instances, common-clock,
    // READ_LATENCY_B=1. addr = {oc_group[3:0], cin[6:0]} unchanged (contract
    // doc S1.1.1, fixed 128 stride) - only the storage/access primitive
    // changes, not the addressing scheme.
    //
    // Contract doc S3 proves (w_idx_next case table) that the two writes in
    // one LOAD beat always target two different taps regardless of whether
    // lane/addr match - so at lane x tap (48-way) granularity, a beat's two
    // writes NEVER target the same bank. No beat buffer, no dual-port, no
    // partial-write/byte-enable merge is needed (contract doc S3/S4 option A).
    localparam WBANK_ADDR_W = $clog2(WBANK_DEPTH);
    localparam WBANK_READ_ADDR_W = $clog2(WBANK_DEPTH / `NPU_PIN);

    // Register one validated weight beat before decoding it onto the 48 BRAM
    // write ports.  The payload/tags are sampled unconditionally so the
    // frame-length/error/phase cone only drives the one-bit valid register,
    // not wide register CEs or every BRAM ENARDEN pin.  A previous command
    // may commit while the next AXIS beat is captured, preserving 1 beat/clk.
    (* max_fanout = 4 *) reg weight_commit_valid;
    reg [3:0]              weight_commit_lane0;
    reg [3:0]              weight_commit_lane1;
    reg [1:0]              weight_commit_tap0;
    reg [1:0]              weight_commit_tap1;
    reg [WBANK_ADDR_W-1:0] weight_commit_addr0;
    reg [WBANK_ADDR_W-1:0] weight_commit_addr1;
    reg [15:0]             weight_commit_data0;
    reg [15:0]             weight_commit_data1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_commit_valid <= 1'b0;
        end
        else if (adapter_clear || !op_active) begin
            weight_commit_valid <= 1'b0;
        end
        else begin
            weight_commit_valid <= load_fire && !load_error &&
                                   (phase == PHASE_LOAD_WEIGHT);
            weight_commit_lane0 <= w_lane0;
            weight_commit_lane1 <= w_lane1;
            weight_commit_tap0  <= w_k_idx;
            weight_commit_tap1  <= w_k_n1;
            weight_commit_addr0 <= w_addr0[WBANK_ADDR_W-1:0];
            weight_commit_addr1 <= w_addr1[WBANK_ADDR_W-1:0];
            weight_commit_data0 <= s_axis_tdata[15:0];
            weight_commit_data1 <= s_axis_tdata[31:16];
        end
    end

    wire [15:0]             wbank_wr_data [0:15][0:2];
    wire                    wbank_wr_en   [0:15][0:2];
    wire [WBANK_ADDR_W-1:0] wbank_wr_addr [0:15][0:2];
    wire [63:0]             wbank_rd_data [0:15][0:2];

    genvar wl, wt;
    generate
        for (wl = 0; wl < 16; wl = wl + 1) begin : GEN_WBANK_LANE
            for (wt = 0; wt < 3; wt = wt + 1) begin : GEN_WBANK_TAP
                // Contract doc S3 invariant: step0 targets (w_lane0,w_k_idx),
                // step1 targets (w_lane1,w_k_n1), and k_idx!=k_n1 always -> at
                // most one of the two OR terms below can be true for any
                // given (wl,wt) in a single beat.
                assign wbank_wr_en[wl][wt] =
                    weight_commit_valid &&
                    (((weight_commit_lane0 == wl[3:0]) &&
                      (weight_commit_tap0  == wt[1:0])) ||
                     ((weight_commit_lane1 == wl[3:0]) &&
                      (weight_commit_tap1  == wt[1:0])));

                assign wbank_wr_data[wl][wt] =
                    ((weight_commit_lane0 == wl[3:0]) &&
                     (weight_commit_tap0  == wt[1:0])) ? weight_commit_data0 :
                                                         weight_commit_data1;

                assign wbank_wr_addr[wl][wt] =
                    ((weight_commit_lane0 == wl[3:0]) &&
                     (weight_commit_tap0  == wt[1:0])) ? weight_commit_addr0 :
                                                         weight_commit_addr1;

                xpm_memory_sdpram #(
                    .ADDR_WIDTH_A        (WBANK_ADDR_W),
                    .ADDR_WIDTH_B        (WBANK_READ_ADDR_W),
                    .AUTO_SLEEP_TIME     (0),
                    .BYTE_WRITE_WIDTH_A  (16),
                    .CASCADE_HEIGHT      (0),
                    .CLOCKING_MODE       ("common_clock"),
                    .ECC_MODE            ("no_ecc"),
                    .MEMORY_INIT_FILE    ("none"),
                    .MEMORY_INIT_PARAM   ("0"),
                    .MEMORY_OPTIMIZATION ("true"),
                    .MEMORY_PRIMITIVE    ("auto"),
                    .MEMORY_SIZE         (WBANK_DEPTH * 16),
                    .MESSAGE_CONTROL     (0),
                    .READ_DATA_WIDTH_B   (64),
                    .READ_LATENCY_B      (1),
                    .READ_RESET_VALUE_B  ("0"),
                    .RST_MODE_A          ("SYNC"),
                    .RST_MODE_B          ("SYNC"),
                    .SIM_ASSERT_CHK      (0),
                    .USE_EMBEDDED_CONSTRAINT (0),
                    .USE_MEM_INIT        (0),
                    .WAKEUP_TIME         ("disable_sleep"),
                    .WRITE_DATA_WIDTH_A  (16),
                    .WRITE_MODE_B        ("no_change")
                ) u_wbank (
                    .clka          (clk),
                    .ena           (1'b1),
                    .wea           (wbank_wr_en[wl][wt]),
                    .addra         (wbank_wr_addr[wl][wt]),
                    .dina          (wbank_wr_data[wl][wt]),
                    .injectsbiterra(1'b0),
                    .injectdbiterra(1'b0),

                    .clkb          (clk),
                    .rstb          (1'b0),
                    .enb           (1'b1),
                    .regceb        (1'b1),
                    .addrb         (weight_rd_addr),
                    .doutb         (wbank_rd_data[wl][wt]),
                    .sleep         (1'b0),
                    .dbiterrb      (),
                    .sbiterrb      ()
                );
            end
        end
    endgenerate

    // ---- read-valid gated stall (Docs/C_WeightBank_16Way_BRAM_Redesign_
    // Contract_v1.md S5, 1st implementation, no predictor). weight_rd_addr
    // comes from conv_compute_ctrl's own REGISTERED issue index (cin_idx_w/
    // oc_group_idx_w), so it only changes on B's own input_fire - meaning it
    // is guaranteed stable across a cycle boundary once B has stalled on it.
    // The xpm_memory_sdpram read port has READ_LATENCY_B=1: wbank_rd_data at
    // cycle N reflects weight_rd_addr as it was at cycle N-1. So the data is
    // valid at cycle N iff the address did NOT change between N-1 and N -
    // i.e. weight_rd_addr(N) == addr_reg (=weight_rd_addr(N-1)). The extra
    // run_first_cycle_done guard exists because addr_reg resets to 0 and
    // weight_rd_addr's very first value (oc_group=0,cin=0) is also 0 - without
    // this guard, cycle 1 of RUN would spuriously read valid=1 before the
    // memory has latched anything for this operation (see contract doc S5
    // timing table T0/T1).
    reg [8:0] wrd_addr_reg;
    reg        run_first_cycle_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wrd_addr_reg         <= 9'd0;
            run_first_cycle_done <= 1'b0;
        end
        else if (adapter_clear || (phase != PHASE_RUN)) begin
            wrd_addr_reg         <= 9'd0;
            run_first_cycle_done <= 1'b0;
        end
        else begin
            wrd_addr_reg         <= weight_rd_addr;
            run_first_cycle_done <= 1'b1;
        end
    end
    wire weight_rd_valid = run_first_cycle_done && (weight_rd_addr == wrd_addr_reg);
    wire weight_rd_valid_eff = USE_SHARED_PARAMS ? i_shared_weight_valid : weight_rd_valid;

    wire [1:0] row_tap1 = read_row_ptr;                                   // t
    wire [1:0] row_tap0 = (read_row_ptr == 2'd0) ? 2'd3 : read_row_ptr - 2'd1; // t-1
    wire [1:0] row_tap2 = (read_row_ptr == 2'd3) ? 2'd0 : read_row_ptr + 2'd1; // t+1

    wire first_window = (compute_t == {`NPU_CFG_T_W{1'b0}});
    wire last_window   = (compute_t == conv_last_t_r);

    // act_row_valid_core: original line-buffer-side readiness (unchanged).
    // Renamed from act_row_valid - the actual i_act_valid/i_weight_valid gate
    // fed to B core is now ANDed with weight_rd_valid below (Docs/
    // C_WeightBank_16Way_BRAM_Redesign_Contract_v1.md S5, 1st implementation:
    // read-valid gated stall, no predictor). weight_rd_valid is declared
    // further down (after weight_rd_addr) but referenced here - legal in
    // Verilog (module-level wire declarations are order-independent).
    wire act_row_valid_core = first_window ? (rows_filled >= {{(`NPU_CFG_T_W-2){1'b0}}, 2'd2}) :
                          last_window  ? (rows_filled == conv_out_len_r) :
                                          (rows_filled >= compute_t + {{(`NPU_CFG_T_W-2){1'b0}}, 2'd2});

    // Request-side row readiness must describe the look-ahead cursor, not
    // the response currently being captured.  This prevents the pipeline
    // from reserving a response for the next time window before its third
    // activation row has arrived.
    wire request_first_window = (request_compute_t_w == {`NPU_CFG_T_W{1'b0}});
    wire request_last_window  = (request_compute_t_w == conv_last_t_r);
    wire request_act_row_valid = request_first_window ?
        (rows_filled >= {{(`NPU_CFG_T_W-2){1'b0}}, 2'd2}) :
        request_last_window ? (rows_filled == conv_out_len_r) :
        (rows_filled >= request_compute_t_w + {{(`NPU_CFG_T_W-2){1'b0}}, 2'd2});

    // capture_arm is the synchronous-memory request token.  The BRAM address
    // is sampled on the request edge and the local operand register consumes
    // the returned payload one clock later, so current-cycle *_rd_valid is
    // deliberately not part of request admission.  Initial group-0 data is
    // primed before RUN; later group bias changes remain explicitly gated at
    // capture_arm_fire below.
    wire act_row_valid = act_row_valid_core;

    wire [`NPU_PIN4_X_BUS_W-1:0] i_x_pin4;
    genvar xpin;
    generate
        for (xpin=0; xpin<`NPU_PIN; xpin=xpin+1) begin : GEN_PIN4_ACT
            assign i_x_pin4[`NPU_PIN4_X_BIT_OFS(xpin,0) +: 16] =
                first_window ? 16'sd0 : line_row_rd[row_tap0][xpin*16 +: 16];
            assign i_x_pin4[`NPU_PIN4_X_BIT_OFS(xpin,1) +: 16] =
                line_row_rd[row_tap1][xpin*16 +: 16];
            assign i_x_pin4[`NPU_PIN4_X_BIT_OFS(xpin,2) +: 16] =
                last_window ? 16'sd0 : line_row_rd[row_tap2][xpin*16 +: 16];
        end
    endgenerate

    assign dbg_step_fire     = step_fire;
    // Debug payload/tag assignments are made beside the local operand buffer
    // below so they describe the transaction actually transferred.
    // dbg_sel_weight is assigned further down, right after sel_weight's own
    // declaration (xvlog SystemVerilog analysis enforces declare-before-use).

    // Last step of the current time window: cin wrapped AND oc_group
    // wrapped on the same accepted step - mirrors conv_compute_ctrl.v's own
    // advance condition exactly (loop order Cin-inner/OC-group/Time-outer),
    // so this is guaranteed to fire on the same cycle B itself advances
    // time_idx internally.
    wire producer_last_step_of_window = capture_arm && current_last_window_step;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_row_ptr  <= 2'd0;
            compute_t     <= {`NPU_CFG_T_W{1'b0}};
            rows_consumed <= {`NPU_CFG_T_W{1'b0}};
            issue_cin_idx_r   <= {`NPU_CFG_C_W{1'b0}};
            issue_group_idx_r <= {`NPU_GROUP_W{1'b0}};
            producer_done_r   <= 1'b0;
        end
        else if (adapter_clear || (phase != PHASE_RUN)) begin
            read_row_ptr  <= 2'd0;
            compute_t     <= {`NPU_CFG_T_W{1'b0}};
            rows_consumed <= {`NPU_CFG_T_W{1'b0}};
            issue_cin_idx_r   <= {`NPU_CFG_C_W{1'b0}};
            issue_group_idx_r <= {`NPU_GROUP_W{1'b0}};
            producer_done_r   <= 1'b0;
        end
        else if (capture_arm) begin
            if ((issue_cin_idx_r + 3'd4) >= cin_r) begin
                issue_cin_idx_r <= {`NPU_CFG_C_W{1'b0}};
                if (issue_group_idx_r == num_oc_group_r - 1'b1)
                    issue_group_idx_r <= {`NPU_GROUP_W{1'b0}};
                else
                    issue_group_idx_r <= issue_group_idx_r + 1'b1;
            end
            else begin
                issue_cin_idx_r <= issue_cin_idx_r + 3'd4;
            end

            if (producer_last_step_of_window) begin
                read_row_ptr  <= (read_row_ptr == 2'd3) ? 2'd0 : read_row_ptr + 2'd1;
                compute_t     <= compute_t + 1'b1;
                rows_consumed <= rows_consumed + 1'b1;
            end

            if (current_final_step)
                producer_done_r <= 1'b1;
        end
    end

    // s_axis_tready: LOAD always accepts one beat/cycle; RUN gated by line
    // buffer room. cfg_ready added (CFG_DERIVE above) - blocks acceptance of
    // any beat (LOAD or RUN) until the 2-stage descriptor-derived register
    // pipeline has settled, so load_fire/run_axis_fire can never sample
    // beats_per_group_m1/total_beats_m1/run_expected_beats_r before they are
    // valid, even if s_axis_tvalid is already asserted the same cycle
    // op_active goes high.
    // PREFETCH_REQ/LATCH are internal, non-AXIS phases.  Qualify RUN ready
    // explicitly so an activation beat cannot handshake and then be dropped
    // while the parameter BRAM outputs are being primed.
    assign s_axis_tready = op_active && cfg_ready && b_rst_n_reg && !i_soft_reset_pulse &&
                            (USE_SHARED_PARAMS ? ((phase == PHASE_RUN) && run_hw_tready) :
                                                 (is_load_phase || ((phase == PHASE_RUN) && run_hw_tready)));

    // ============================================================
    // Weight/bias/rshift combinational selection (design doc S1.1.1/S2) -
    // issue-side for weight/bias/activation, result-side for rshift
    // (frozen spec S5 - rshift MUST be result-aligned).
    // ============================================================
    wire [`NPU_GROUP_W-1:0] result_group_idx_w;
    assign result_group_idx_w = i_b_result_group_idx;

    // ---- weight readout: reassemble sel_weight from the 48 lane x tap
    // xpm_memory_sdpram outputs (GEN_WBANK_LANE/GEN_WBANK_TAP above) instead
    // of a single behavioral array read. NPU_WEIGHT_BIT_OFS packing/B core
    // interface unchanged (contract doc S3 option A). ----
    wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] sel_weight;
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : WBANK_READ
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,0,0) +: 16] = wbank_rd_data[gi][0][15:0];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,0,1) +: 16] = wbank_rd_data[gi][1][15:0];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,0,2) +: 16] = wbank_rd_data[gi][2][15:0];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,1,0) +: 16] = wbank_rd_data[gi][0][31:16];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,1,1) +: 16] = wbank_rd_data[gi][1][31:16];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,1,2) +: 16] = wbank_rd_data[gi][2][31:16];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,2,0) +: 16] = wbank_rd_data[gi][0][47:32];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,2,1) +: 16] = wbank_rd_data[gi][1][47:32];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,2,2) +: 16] = wbank_rd_data[gi][2][47:32];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,3,0) +: 16] = wbank_rd_data[gi][0][63:48];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,3,1) +: 16] = wbank_rd_data[gi][1][63:48];
            assign sel_weight[`NPU_PIN4_WEIGHT_BIT_OFS(gi,3,2) +: 16] = wbank_rd_data[gi][2][63:48];
        end
    endgenerate

    wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] sel_weight_eff = USE_SHARED_PARAMS ? i_shared_weight : sel_weight;

    // Bias is issue-aligned against oc_group_idx_w.  Rshift is result-aligned
    // against result_group_idx_w.  Each payload lives in independent
    // synchronous BRAM banks and is copied into a small cache on a group
    // boundary.  A bias miss stalls operand capture; a rshift miss deasserts
    // o_b_rshift_valid so the shared core holds its result before postprocess.
    wire [`NPU_BIAS_BUS_W-1:0]   sel_bias;
    wire [`NPU_RSHIFT_BUS_W-1:0] sel_rshift;

    reg [47:0] bias_cache_lane [0:15];
    reg [31:0] rshift_cache_word [0:3];
    reg [PARAM_GROUP_AW-1:0] bias_cache_group, rshift_cache_group;
    reg [PARAM_GROUP_AW-1:0] bias_prefetch_group, rshift_prefetch_group;
    reg bias_cache_valid, rshift_cache_valid;
    reg bias_prefetch_pending, rshift_prefetch_pending;

    wire [PARAM_GROUP_AW-1:0] issue_group = oc_group_idx_w[PARAM_GROUP_AW-1:0];
    wire [PARAM_GROUP_AW-1:0] result_group = result_group_idx_w[PARAM_GROUP_AW-1:0];
    wire bias_cache_match = bias_cache_valid && (bias_cache_group == issue_group);
    wire rshift_cache_match = rshift_cache_valid && (rshift_cache_group == result_group);
    wire initial_prefetch_req = (phase == PHASE_PREFETCH_REQ);
    wire bias_runtime_req = (phase == PHASE_RUN) && !bias_cache_match && !bias_prefetch_pending;
    wire rshift_runtime_req = (phase == PHASE_RUN) && !rshift_cache_match && !rshift_prefetch_pending;
    wire bias_mem_rd_en = initial_prefetch_req || bias_runtime_req;
    wire rshift_mem_rd_en = initial_prefetch_req || rshift_runtime_req;
    wire [PARAM_GROUP_AW-1:0] bias_mem_rd_addr = initial_prefetch_req ? {PARAM_GROUP_AW{1'b0}} : issue_group;
    wire [PARAM_GROUP_AW-1:0] rshift_mem_rd_addr = initial_prefetch_req ? {PARAM_GROUP_AW{1'b0}} : result_group;

    wire bias_accept_commit = load_fire && !load_error &&
                              (phase == PHASE_LOAD_BIAS) && ldw_bias_half;
    wire rshift_accept_commit = load_fire && !load_error && (phase == PHASE_LOAD_RSHIFT);

    // Validate/capture a complete 48-bit bias word first, then commit it to
    // BRAM one cycle later.  This keeps the expected-frame-length comparator
    // off all sixteen BRAM write-enable pins while preserving one accepted
    // AXIS beat per cycle (capture and prior commit may overlap).
    // Rshift uses the same capture/commit split as bias.  The registered
    // command removes frame-length/TLAST validation from the four BRAM WE
    // cones.  PHASE_RSHIFT_DRAIN above guarantees that the final command is
    // physically committed before the first synchronous prefetch request.
    reg                         rshift_commit_valid;
    reg [3:0]                   rshift_commit_word_onehot;
    reg [PARAM_GROUP_AW-1:0]    rshift_commit_group;
    reg [31:0]                  rshift_commit_data;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rshift_commit_valid       <= 1'b0;
            rshift_commit_word_onehot <= 4'd0;
            rshift_commit_group       <= {PARAM_GROUP_AW{1'b0}};
            rshift_commit_data        <= 32'd0;
        end else if (adapter_clear || !op_active) begin
            rshift_commit_valid       <= 1'b0;
            rshift_commit_word_onehot <= 4'd0;
        end else begin
            rshift_commit_valid <= rshift_accept_commit;
            if (rshift_accept_commit) begin
                rshift_commit_word_onehot <= ldw_rshift_word_onehot;
                rshift_commit_group       <= group_idx[PARAM_GROUP_AW-1:0];
                rshift_commit_data        <= s_axis_tdata;
            end
        end
    end

    genvar pbl, prw;
    generate
        for (pbl = 0; pbl < 16; pbl = pbl + 1) begin : GEN_CONVN_BIAS_LANE
            // Each BRAM receives a physically local command register.  This
            // duplicates only the small LOAD commit state and eliminates the
            // global group/data/WE nets that previously crossed all 16 banks.
            reg                      commit_valid;
            reg [PARAM_GROUP_AW-1:0] commit_group;
            reg [47:0]               commit_data;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    commit_valid <= 1'b0;
                    commit_group <= {PARAM_GROUP_AW{1'b0}};
                    commit_data  <= 48'd0;
                end
                else if (adapter_clear || !op_active) begin
                    commit_valid <= 1'b0;
                end
                else begin
                    commit_valid <= bias_accept_commit && ldw_bias_lane_onehot[pbl];
                    if (bias_accept_commit && ldw_bias_lane_onehot[pbl]) begin
                        commit_group <= group_idx[PARAM_GROUP_AW-1:0];
                        commit_data  <= {s_axis_tdata[15:0], bias_staging};
                    end
                end
            end
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A(PARAM_GROUP_AW), .ADDR_WIDTH_B(PARAM_GROUP_AW),
                .AUTO_SLEEP_TIME(0), .BYTE_WRITE_WIDTH_A(48), .CASCADE_HEIGHT(0),
                .CLOCKING_MODE("common_clock"), .ECC_MODE("no_ecc"),
                .MEMORY_INIT_FILE("none"), .MEMORY_INIT_PARAM("0"),
                .MEMORY_OPTIMIZATION("true"), .MEMORY_PRIMITIVE("block"),
                .MEMORY_SIZE(MAX_OC_GROUP*48), .MESSAGE_CONTROL(0),
                .READ_DATA_WIDTH_B(48), .READ_LATENCY_B(1),
                .READ_RESET_VALUE_B("0"), .RST_MODE_A("SYNC"), .RST_MODE_B("SYNC"),
                .SIM_ASSERT_CHK(0), .USE_EMBEDDED_CONSTRAINT(0), .USE_MEM_INIT(0),
                .WAKEUP_TIME("disable_sleep"), .WRITE_DATA_WIDTH_A(48),
                .WRITE_MODE_B("no_change")
            ) u_bias_bank (
                .clka(clk), .ena(1'b1),
                .wea(commit_valid),
                .addra(commit_group),
                .dina(commit_data),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .clkb(clk), .rstb(1'b0), .enb(bias_mem_rd_en),
                .regceb(1'b1), .addrb(bias_mem_rd_addr),
                .doutb(bias_bank_rd[pbl]), .sleep(1'b0),
                .dbiterrb(), .sbiterrb()
            );
            assign sel_bias[pbl*48 +: 48] = bias_cache_lane[pbl];
        end
        for (prw = 0; prw < 4; prw = prw + 1) begin : GEN_CONVN_RSHIFT_WORD
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A(PARAM_GROUP_AW), .ADDR_WIDTH_B(PARAM_GROUP_AW),
                .AUTO_SLEEP_TIME(0), .BYTE_WRITE_WIDTH_A(32), .CASCADE_HEIGHT(0),
                .CLOCKING_MODE("common_clock"), .ECC_MODE("no_ecc"),
                .MEMORY_INIT_FILE("none"), .MEMORY_INIT_PARAM("0"),
                .MEMORY_OPTIMIZATION("true"), .MEMORY_PRIMITIVE("block"),
                .MEMORY_SIZE(MAX_OC_GROUP*32), .MESSAGE_CONTROL(0),
                .READ_DATA_WIDTH_B(32), .READ_LATENCY_B(1),
                .READ_RESET_VALUE_B("0"), .RST_MODE_A("SYNC"), .RST_MODE_B("SYNC"),
                .SIM_ASSERT_CHK(0), .USE_EMBEDDED_CONSTRAINT(0), .USE_MEM_INIT(0),
                .WAKEUP_TIME("disable_sleep"), .WRITE_DATA_WIDTH_A(32),
                .WRITE_MODE_B("no_change")
            ) u_rshift_bank (
                .clka(clk), .ena(1'b1),
                .wea(rshift_commit_valid && rshift_commit_word_onehot[prw]),
                .addra(rshift_commit_group), .dina(rshift_commit_data),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .clkb(clk), .rstb(1'b0), .enb(rshift_mem_rd_en),
                .regceb(1'b1), .addrb(rshift_mem_rd_addr),
                .doutb(rshift_bank_rd[prw]), .sleep(1'b0),
                .dbiterrb(), .sbiterrb()
            );
            assign sel_rshift[prw*32 +: 32] = rshift_cache_word[prw];
        end
    endgenerate

    integer bcl, rcl;
    always @(posedge clk) begin
        // Cache validity/group state below remains the sole authority for
        // observability.  Capturing BRAM outputs unconditionally removes the
        // wide pending-to-CE fanout without exposing stale data.
        for (bcl = 0; bcl < 16; bcl = bcl + 1)
            bias_cache_lane[bcl] <= bias_bank_rd[bcl];
        for (rcl = 0; rcl < 4; rcl = rcl + 1)
            rshift_cache_word[rcl] <= rshift_bank_rd[rcl];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_cache_valid       <= 1'b0;
            rshift_cache_valid     <= 1'b0;
            bias_prefetch_pending  <= 1'b0;
            rshift_prefetch_pending <= 1'b0;
            bias_cache_group       <= {PARAM_GROUP_AW{1'b0}};
            rshift_cache_group     <= {PARAM_GROUP_AW{1'b0}};
            bias_prefetch_group    <= {PARAM_GROUP_AW{1'b0}};
            rshift_prefetch_group  <= {PARAM_GROUP_AW{1'b0}};
        end
        else if (adapter_clear || !op_active) begin
            bias_cache_valid       <= 1'b0;
            rshift_cache_valid     <= 1'b0;
            bias_prefetch_pending  <= 1'b0;
            rshift_prefetch_pending <= 1'b0;
        end
        else begin
            if (initial_prefetch_req) begin
                bias_cache_valid        <= 1'b0;
                rshift_cache_valid      <= 1'b0;
                bias_prefetch_group     <= {PARAM_GROUP_AW{1'b0}};
                rshift_prefetch_group   <= {PARAM_GROUP_AW{1'b0}};
                bias_prefetch_pending   <= 1'b1;
                rshift_prefetch_pending <= 1'b1;
            end
            else begin
                if (bias_runtime_req) begin
                    bias_cache_valid      <= 1'b0;
                    bias_prefetch_group   <= issue_group;
                    bias_prefetch_pending <= 1'b1;
                end
                else if (bias_prefetch_pending) begin
                    bias_cache_group      <= bias_prefetch_group;
                    bias_cache_valid      <= 1'b1;
                    bias_prefetch_pending <= 1'b0;
                end

                if (rshift_runtime_req) begin
                    rshift_cache_valid      <= 1'b0;
                    rshift_prefetch_group   <= result_group;
                    rshift_prefetch_pending <= 1'b1;
                end
                else if (rshift_prefetch_pending) begin
                    rshift_cache_group      <= rshift_prefetch_group;
                    rshift_cache_valid      <= 1'b1;
                    rshift_prefetch_pending <= 1'b0;
                end
            end
        end
    end

    // ============================================================
    // Output side: drain one 256-bit result beat into 8 AXIS words. Last-
    // group decision uses the adapter's own output_group_count against
    // last_output_group_r (manifest-authoritative, see i_last_output_group),
    // never a B result-side index (design doc S1.3 principle, unchanged from
    // Conv1).
    // ============================================================
    localparam DASM_IDLE  = 1'b0;
    localparam DASM_DRAIN = 1'b1;

    reg                          dasm_state;
    reg [`NPU_OUT_BUS_W-1:0]     dasm_shift_reg;
    reg [2:0]                    dasm_word_idx;
    reg                          dasm_is_last_group;
    reg [31:0]                   output_group_count;

    wire core_out_valid_w, core_out_ready_w;
    wire [`NPU_OUT_BUS_W-1:0] core_out_data_w;

    assign core_out_valid_w = i_b_out_valid;
    assign core_out_data_w  = i_b_out_data;

    wire core_out_fire = core_out_valid_w && core_out_ready_w;
    wire m_axis_fire    = m_axis_tvalid && m_axis_tready;

    assign core_out_ready_w = (dasm_state == DASM_IDLE);
    assign o_b_out_ready    = core_out_ready_w;
    assign m_axis_tvalid    = (dasm_state == DASM_DRAIN);
    assign m_axis_tdata     = dasm_shift_reg[dasm_word_idx*32 +: 32];
    assign m_axis_tkeep     = 4'hF;
    assign m_axis_tlast     = (dasm_state == DASM_DRAIN) && (dasm_word_idx == 3'd7) && dasm_is_last_group;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dasm_state          <= DASM_IDLE;
            dasm_shift_reg      <= {`NPU_OUT_BUS_W{1'b0}};
            dasm_word_idx       <= 3'd0;
            dasm_is_last_group  <= 1'b0;
            output_group_count  <= 32'd0;
        end
        else if (adapter_clear || !op_active) begin
            dasm_state          <= DASM_IDLE;
            dasm_shift_reg      <= {`NPU_OUT_BUS_W{1'b0}};
            dasm_word_idx       <= 3'd0;
            dasm_is_last_group  <= 1'b0;
            output_group_count  <= 32'd0;
        end
        else begin
            case (dasm_state)
                DASM_IDLE: begin
                    if (core_out_fire) begin
                        dasm_shift_reg      <= core_out_data_w;
                        dasm_is_last_group  <= (output_group_count == last_output_group_r);
                        dasm_word_idx       <= 3'd0;
                        dasm_state          <= DASM_DRAIN;
                        output_group_count  <= output_group_count + 1'b1;
                    end
                end
                DASM_DRAIN: begin
                    if (m_axis_fire) begin
                        if (dasm_word_idx == 3'd7) begin
                            dasm_word_idx <= 3'd0;
                            dasm_state    <= DASM_IDLE;
                        end
                        else begin
                            dasm_word_idx <= dasm_word_idx + 3'd1;
                        end
                    end
                end
                default: dasm_state <= DASM_IDLE;
            endcase
        end
    end

    assign o_done = m_axis_fire && m_axis_tlast;

    // ============================================================
    // Shared B-core boundary - CONV mode, Cin>1.  The unified top owns the
    // sole b_compute_top_16 instance and supplies these returned signals.
    // ============================================================
    wire b_core_busy_w, b_core_done_w;
    wire act_ready_w;
    assign b_core_busy_w = i_b_busy;
    assign b_core_done_w = i_b_done;
    assign act_ready_w   = i_b_act_ready;

    // C-side independent evaluation - Docs/
    // C_BCore_DSP_Cascade_Timing_Investigation_v1.md S11 (adapter->B-core
    // operand issue buffer). NOT an official B-core/adapter change - lives
    // only in Vivado/rtl_dev/bcore_dsp_pipeline_eval_c_side/, isolated from
    // the real b_core_configtiming_fix dev copy and any packaged IP.
    //
    // Breaks the fully-combinational read_row_ptr -> row_tap mux ->
    // line_buf read -> i_x2 -> B core's s0_x2_reg path (raw-route
    // WNS=-0.018ns, S10/S11 OOC result) by registering one full operand
    // issue bundle (x0/x1/x2 + weight + bias, NOT rshift - rshift is an
    // unrelated output-side signal) between the adapter's combinational
    // read and B core's own s0 input.
    //
    // Two handshakes are deliberately kept separate (per explicit review
    // correction - a single act_ready_w-derived buf_ready would let the
    // SAME (not-yet-advanced) cin_idx_w/oc_group_idx_w operand be captured
    // a second time on the very edge B dequeues it, since cin_idx_w/
    // oc_group_idx_w are B's own registered counters and only advance
    // AFTER this edge):
    //   core_dequeue_fire = opbuf_valid && act_ready_w   - B genuinely
    //     accepts the buffered operand this edge; this is the ONLY event
    //     that may advance read_row_ptr/compute_t/rows_consumed, since
    //     those must stay in lockstep with B's own accepted-step index
    //     (weight_rd_addr is already keyed off cin_idx_w/oc_group_idx_w -
    //     see the comment above GEN_WBANK_LANE).
    //   opbuf_enqueue_fire = act_row_valid && !opbuf_valid - only capture a
    //     fresh operand when the buffer is genuinely empty (gated on the
    //     PRE-edge opbuf_valid, not a same-edge-freed buf_ready), so a
    //     dequeue and a re-capture of the same stale index can never
    //     collide on one edge.
    //
    // Known cost (explicitly NOT hidden): this design cannot enqueue and
    // dequeue on the same edge, so it does not sustain 1 step/clock the
    // way the old direct connection could - throughput must be measured,
    // not assumed.
    //
    // ---- S11 follow-up: 2-stage capture + lane-local payload storage ----
    // The 1-stage version above hit a NEW worst-setup path (raw-route
    // WNS=+0.429ns after the bank-staging fix, still short of the +1.0ns
    // target): compute_t_reg -> opbuf_x0_reg[*]/opbuf_bias_reg[*] CE - the
    // compute_t->act_row_valid combinational cone (deep logic, not just
    // fanout) was driving the CE of hundreds of opbuf_* bits directly.
    //
    // Fix: split into two cycles.
    //   Cycle N   (capture_arm_fire = act_row_valid && !opbuf_valid &&
    //              !capture_arm): the compute_t->act_row_valid cone only
    //              has to reach ONE bit (capture_arm), not hundreds.
    //   Cycle N+1 (gated by the now-CLEAN, already-registered capture_arm,
    //              not by live combinational logic): the wide payload
    //              capture happens here. i_x0/i_x1/i_x2/sel_weight/
    //              sel_bias are still combinationally valid/unchanged from
    //              cycle N, because nothing advances read_row_ptr/
    //              cin_idx_w/oc_group_idx_w except core_dequeue_fire,
    //              which cannot fire while opbuf_valid is 0 (nothing to
    //              dequeue) - so re-reading them one cycle later is safe.
    //   Cycle N+2 (core_dequeue_fire = opbuf_valid && act_ready_w): B
    //              dequeues the payload - unchanged from before.
    //
    // weight/bias are ALSO split from one monolithic 768-bit bus register
    // into 16 lane-local 48-bit registers (opbuf_weight_lane/opbuf_bias_
    // lane), so capture_arm's fanout is distributed across 16 independent
    // per-lane generate instances rather than one 768-bit-wide register,
    // letting placement cluster each lane's copy near its own consumer.
    // Reassembled into the flat bus b_compute_top_16 expects via a
    // read-only generate (mirrors the S11 bank-staging read-side pattern -
    // plain wire reassembly has no "driver ambiguity" risk, only writes
    // do).
    // Two local entries plus the one outstanding synchronous-memory token
    // form a credit-safe response queue.  A surprise downstream stall can
    // therefore consume the reserved token without overwriting the resident
    // operand, while the normal path still captures and dequeues every edge.
    reg [1:0] opbuf_count;
    reg       opbuf_rd_ptr;
    reg       opbuf_wr_ptr;
    reg [`NPU_PIN4_X_BUS_W-1:0] opbuf_x_mem [0:1];
    reg [`NPU_CFG_T_W-1:0] opbuf_time_mem [0:1];
    reg [`NPU_GROUP_W-1:0] opbuf_group_mem [0:1];
    reg [`NPU_CFG_C_W-1:0] opbuf_cin_mem [0:1];
    reg [`NPU_WEIGHT_W*`NPU_PIN*`NPU_TAPS-1:0] opbuf_weight_lane0 [0:`NPU_LANES-1];
    reg [`NPU_WEIGHT_W*`NPU_PIN*`NPU_TAPS-1:0] opbuf_weight_lane1 [0:`NPU_LANES-1];
    reg [`NPU_ACC_W-1:0] opbuf_bias_lane0 [0:`NPU_LANES-1];
    reg [`NPU_ACC_W-1:0] opbuf_bias_lane1 [0:`NPU_LANES-1];

    wire opbuf_valid = (opbuf_count != 2'd0);
    wire [`NPU_PIN4_X_BUS_W-1:0] opbuf_x = opbuf_rd_ptr ? opbuf_x_mem[1] : opbuf_x_mem[0];
    wire [`NPU_CFG_T_W-1:0] opbuf_time_idx = opbuf_rd_ptr ? opbuf_time_mem[1] : opbuf_time_mem[0];
    wire [`NPU_GROUP_W-1:0] opbuf_group_idx = opbuf_rd_ptr ? opbuf_group_mem[1] : opbuf_group_mem[0];
    wire [`NPU_CFG_C_W-1:0] opbuf_cin_idx = opbuf_rd_ptr ? opbuf_cin_mem[1] : opbuf_cin_mem[0];

    wire core_dequeue_fire = opbuf_valid && act_ready_w;
    // Bias is consumed on the issue side.  Never arm a new operand until the
    // cache for B's current output-channel group is resident.
    // Shared ConvN bias comes from the parameter BRAM's one-cycle prefetch
    // output.  The producer request edge launches that read and capture_arm
    // consumes it on the following edge, so no cache-valid bubble is needed
    // at an output-group boundary.  Standalone mode retains its local cache.
    wire bias_cache_match_eff = USE_SHARED_PARAMS ? 1'b1 : bias_cache_match;
    wire [`NPU_BIAS_BUS_W-1:0] sel_bias_eff = USE_SHARED_PARAMS ? i_shared_bias : sel_bias;
    wire [`NPU_RSHIFT_BUS_W-1:0] sel_rshift_eff = USE_SHARED_PARAMS ? i_shared_rshift : sel_rshift;
    // Count the response captured on this edge and the head transferred on
    // this edge before reserving another one-cycle BRAM response.  At most
    // two resident entries plus one explicitly reserved response exist.
    wire [2:0] opbuf_count_after_edge =
        {1'b0,opbuf_count} + (capture_arm ? 3'd1 : 3'd0) -
        (core_dequeue_fire ? 3'd1 : 3'd0);
    wire request_exists = !producer_done_r &&
        ((USE_SHARED_PARAMS != 0) ? (!capture_arm || !current_final_step) : !capture_arm);
    wire capture_arm_fire = request_exists && request_act_row_valid &&
                            bias_cache_match_eff && (opbuf_count_after_edge < 3'd2);

    // capture_arm is both the one-cycle memory-response token and the wide
    // payload write enable.  In the steady state it remains asserted while
    // the request address advances to the following operand every clock.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            capture_arm <= 1'b0;
        else if (adapter_clear || (phase != PHASE_RUN))
            capture_arm <= 1'b0;
        else
            capture_arm <= capture_arm_fire;
    end

    // Response FIFO control and narrow payload/tags.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            opbuf_count  <= 2'd0;
            opbuf_rd_ptr <= 1'b0;
            opbuf_wr_ptr <= 1'b0;
        end
        else if (adapter_clear || (phase != PHASE_RUN)) begin
            opbuf_count  <= 2'd0;
            opbuf_rd_ptr <= 1'b0;
            opbuf_wr_ptr <= 1'b0;
        end
        else begin
            case ({capture_arm,core_dequeue_fire})
                2'b10: opbuf_count <= opbuf_count + 1'b1;
                2'b01: opbuf_count <= opbuf_count - 1'b1;
                default: opbuf_count <= opbuf_count;
            endcase

            if (capture_arm) begin
                opbuf_x_mem[opbuf_wr_ptr]     <= i_x_pin4;
                opbuf_time_mem[opbuf_wr_ptr]  <= compute_t;
                opbuf_group_mem[opbuf_wr_ptr] <= issue_group_idx_r;
                opbuf_cin_mem[opbuf_wr_ptr]   <= issue_cin_idx_r;
                opbuf_wr_ptr                  <= ~opbuf_wr_ptr;
            end

            if (core_dequeue_fire)
                opbuf_rd_ptr <= ~opbuf_rd_ptr;
        end
    end

    genvar opl;
    generate
        for (opl = 0; opl < `NPU_LANES; opl = opl + 1) begin : GEN_OPBUF_LANE
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    opbuf_weight_lane0[opl] <= {(`NPU_PIN*`NPU_TAPS*`NPU_WEIGHT_W){1'b0}};
                    opbuf_weight_lane1[opl] <= {(`NPU_PIN*`NPU_TAPS*`NPU_WEIGHT_W){1'b0}};
                    opbuf_bias_lane0[opl] <= {`NPU_ACC_W{1'b0}};
                    opbuf_bias_lane1[opl] <= {`NPU_ACC_W{1'b0}};
                end
                else if (adapter_clear || (phase != PHASE_RUN)) begin
                    opbuf_weight_lane0[opl] <= {(`NPU_PIN*`NPU_TAPS*`NPU_WEIGHT_W){1'b0}};
                    opbuf_weight_lane1[opl] <= {(`NPU_PIN*`NPU_TAPS*`NPU_WEIGHT_W){1'b0}};
                    opbuf_bias_lane0[opl] <= {`NPU_ACC_W{1'b0}};
                    opbuf_bias_lane1[opl] <= {`NPU_ACC_W{1'b0}};
                end
                else if (capture_arm) begin
                    if (opbuf_wr_ptr) begin
                        opbuf_weight_lane1[opl] <=
                            sel_weight_eff[(opl*`NPU_PIN*`NPU_TAPS)*`NPU_WEIGHT_W +: (`NPU_PIN*`NPU_TAPS*`NPU_WEIGHT_W)];
                        opbuf_bias_lane1[opl] <=
                            sel_bias_eff[opl*`NPU_ACC_W +: `NPU_ACC_W];
                    end else begin
                        opbuf_weight_lane0[opl] <=
                            sel_weight_eff[(opl*`NPU_PIN*`NPU_TAPS)*`NPU_WEIGHT_W +: (`NPU_PIN*`NPU_TAPS*`NPU_WEIGHT_W)];
                        opbuf_bias_lane0[opl] <=
                            sel_bias_eff[opl*`NPU_ACC_W +: `NPU_ACC_W];
                    end
                end
            end
        end
    endgenerate

    // Read-only reassembly into the flat buses b_compute_top_16 expects -
    // plain wire, no driver ambiguity (mirrors the S11 bank-staging
    // read-side fix's reasoning).
    wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] opbuf_weight;
    wire [`NPU_BIAS_BUS_W-1:0]   opbuf_bias;
    genvar opl2;
    generate
        for (opl2 = 0; opl2 < `NPU_LANES; opl2 = opl2 + 1) begin : GEN_OPBUF_LANE_READ
            assign opbuf_weight[(opl2*`NPU_PIN*`NPU_TAPS)*`NPU_WEIGHT_W +: (`NPU_PIN*`NPU_TAPS*`NPU_WEIGHT_W)] =
                opbuf_rd_ptr ? opbuf_weight_lane1[opl2] : opbuf_weight_lane0[opl2];
            assign opbuf_bias[opl2*`NPU_ACC_W +: `NPU_ACC_W] =
                opbuf_rd_ptr ? opbuf_bias_lane1[opl2] : opbuf_bias_lane0[opl2];
            assign dbg_sel_weight[`NPU_WEIGHT_BIT_OFS(opl2,0) +: 16] =
                opbuf_weight[`NPU_PIN4_WEIGHT_BIT_OFS(opl2,0,0) +: 16];
            assign dbg_sel_weight[`NPU_WEIGHT_BIT_OFS(opl2,1) +: 16] =
                opbuf_weight[`NPU_PIN4_WEIGHT_BIT_OFS(opl2,0,1) +: 16];
            assign dbg_sel_weight[`NPU_WEIGHT_BIT_OFS(opl2,2) +: 16] =
                opbuf_weight[`NPU_PIN4_WEIGHT_BIT_OFS(opl2,0,2) +: 16];
        end
    endgenerate

    assign dbg_time_idx     = opbuf_time_idx;
    assign dbg_oc_group_idx = {{(`NPU_CFG_C_W-`NPU_GROUP_W){1'b0}}, opbuf_group_idx};
    assign dbg_cin_idx      = opbuf_cin_idx;
    assign dbg_x0           = opbuf_x[`NPU_PIN4_X_BIT_OFS(0,0) +: 16];
    assign dbg_x1           = opbuf_x[`NPU_PIN4_X_BIT_OFS(0,1) +: 16];
    assign dbg_x2           = opbuf_x[`NPU_PIN4_X_BIT_OFS(0,2) +: 16];

    assign step_fire = core_dequeue_fire;

    // Shared B-core boundary.  All values are exactly the signals that the
    // former private instance consumed; this frontend does not choose an
    // operation mode or instantiate compute hardware.
    assign o_b_start             = b_start_pulse_reg;
    assign o_b_pool_enable       = pool_enable_r;
    assign o_b_conv_cin          = cin_r;
    assign o_b_conv_cout         = cout_r;
    assign o_b_conv_out_len      = conv_out_len_r;
    assign o_b_conv_post_out_len = out_len_r;
    assign o_b_act_valid         = opbuf_valid;
    assign o_b_x                 = opbuf_x;
    assign o_b_weight_valid      = opbuf_valid;
    assign o_b_weight            = opbuf_weight;
    assign o_b_bias              = opbuf_bias;
    assign o_b_rshift            = sel_rshift_eff;
    // Rshift is consumed on the result side, so its validity must follow
    // i_b_result_group_idx rather than the issue-side operand buffer.
    assign o_b_rshift_valid      = (phase == PHASE_RUN) &&
                                    (USE_SHARED_PARAMS ? i_shared_rshift_valid : rshift_cache_match);

    // The frozen adapter contract uses act_ready as the paired
    // activation/weight acceptance condition.  Keep the unused ready port
    // explicit so a later contract assertion can require equality.
    wire unused_b_weight_ready = i_b_weight_ready;

endmodule
