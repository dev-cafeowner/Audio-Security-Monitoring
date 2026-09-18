`timescale 1ns / 1ps
`include "npu_defs.vh"

//////////////////////////////////////////////////////////////////////////////
// Unified-frontend extraction of the historical Conv1 adapter.  Local LOAD,
// windowing, and AXIS behavior are retained, while the B-core ports are now
// explicit: neural_processing_unit_unified_2_0 owns the sole shared core.
//
// Conv1 layer constants (design doc S0): Cin=1, Cout=64 (4 OC groups of 16),
// K=3, S=3, P=1, T_out=106667. Weight=384B, Bias(DDR container)=512B,
// Rshift=64B - all derived/verified in the design doc, not re-derived here.
//
// CSR.START (i_start_pulse) and the shared B-core i_start are DIFFERENT instants (design
// doc S4/S5): an accepted CSR.START moves this adapter into a 3-frame LOAD
// sequence (weight -> bias -> rshift, each PS-driven MM2S transfers waited on
// individually by the driver - this adapter has no AXI-master port and never
// initiates its own DDR reads). Only once all three frames are validated does
// this adapter pulse B's i_start itself and enter RUN.
//
// LOAD framing (design doc S8.2): every LOAD beat MUST have tkeep=4'hF (all
// three frame sizes are exact 4-byte multiples) - unlike RUN's activation
// stream, LOAD never allows a 4'h3 partial beat. A bad tkeep, an early tlast,
// or a missing tlast at the expected final beat aborts the whole operation via
// a single-cycle o_error_pulse/o_error_code (LOAD_TKEEP=0x10/LOAD_TLAST=0x11/
// LOAD_LENGTH=0x12) into npu_csr_axi_lite's backend error port - the AXI-Lite
// register map itself is unchanged (see the Conv1-only copy of that file).
//
// Activation tap generation (design doc S3): S00_AXIS is 32-bit (2 halfwords/
// beat) but a Conv1 window consumes 3 halfwords (K=3,S=3 non-overlapping), so
// AXIS beat boundaries and tap boundaries are permanently out of phase. A
// 4-deep halfword FIFO absorbs the leftover halfword across beats. t=0 is a
// one-time special case (left-pad tap0=0, only 2 real samples consumed);
// every later time_idx consumes exactly 3. Taps are held stable across the 4
// consecutive accepted CONV steps (one per OC group) that share a time_idx,
// and refilled only when o_conv_oc_group_idx wraps 3->0.
//
// Weight/bias/rshift local buffers (design doc S1/S2): weight and rshift load
// as a straight contiguous copy (DDR canonical order already matches B's
// physical bus order for IC=1); bias needs a 16-way gather+truncate (64-bit
// DDR container -> 48-bit lane) since 64 doesn't divide 48. All three fully
// preload into small register files before RUN (4 OC groups is tiny) and are
// then indexed purely combinationally - weight/bias by the issue-side
// o_conv_oc_group_idx, rshift by the result-side o_result_group_idx (frozen
// spec S5 - rshift MUST be result-aligned, not issue-side).
//
// Output framing (design doc S8.3): does NOT use o_result_group_idx/
// o_result_time_idx to detect the final output beat (those are pipeline-
// timing signals, not guaranteed synchronized with this adapter's own AXIS
// serializer position). Instead a private output_group_count (width/limit
// derived from the CONV1_OUT_LEN parameter - 4*CONV1_OUT_LEN wide-output-
// groups total) increments once per core_out_fire and marks group
// 4*CONV1_OUT_LEN-1 (0-indexed) as the last one - TLAST/wrapper DONE fire
// only on that group's 8th AXIS word.
//
// CONV1_OUT_LEN is a module parameter (default 106667, the real Conv1
// layer), not a hardcoded constant - xsim instantiates reduced values (8, 9)
// for the design doc S6.2 partial-beat/clean smoke tests; LAST_OUTPUT_GROUP,
// RUN_EXPECTED_SAMPLES and their counter widths are all derived from it.
//
// RUN-phase framing (design doc S6.2/S8.2 P2 fix): the activation stream
// gets the same class of validation as LOAD - tkeep must be 4'hF on every
// beat except the very last (which may be 4'hF or 4'h3), tlast must land
// exactly on the beat that completes the expected 3*CONV1_OUT_LEN-1 real
// samples (not earlier, not missing), and any violation aborts via the same
// o_error_pulse/o_error_code path as a malformed LOAD frame (codes 0x13-
// 0x15, distinct from LOAD's 0x10-0x12).
//
// SOFT_RESET / b_rst_n_reg / adapter_clear / op_active: identical pattern to
// npu_global_backend_adapter.v (registered one-cycle-low pulse on the B
// core's rst_n, synchronously clearing every FSM in this file too).
//////////////////////////////////////////////////////////////////////////////

module npu_conv1_frontend #(
    // Exposed so xsim can instantiate a reduced T_out (design doc S6.2:
    // T_out=9 clean smoke test, T_out=8 partial-beat regression) instead of
    // the full 106,667 - LAST_OUTPUT_GROUP/RUN_EXPECTED_SAMPLES and their
    // counter widths are all derived from this, not hardcoded (P1 fix).
    parameter CONV1_OUT_LEN = 106667,
    parameter USE_SHARED_PARAMS = 0
) (
    input  wire        clk,
    input  wire        rst_n,

    // ---- npu_csr_axi_lite backend handshake ----
    input  wire        i_start_pulse,
    input  wire        i_soft_reset_pulse,
    output wire        o_busy,
    output wire        o_done,
    output wire        o_error_pulse,
    output wire [7:0]  o_error_code,

    // ---- S00_AXIS: DMA MM2S -> here (LOAD frames, then activation input) ----
    output wire        s_axis_tready,
    input  wire [31:0] s_axis_tdata,
    input  wire [3:0]  s_axis_tkeep,
    input  wire        s_axis_tlast,
    input  wire        s_axis_tvalid,

    // ---- M00_AXIS: here -> DMA S2MM (Conv1 result, driver targets ACT_A) ----
    output wire        m_axis_tvalid,
    output wire [31:0] m_axis_tdata,
    output wire [3:0]  m_axis_tkeep,
    output wire        m_axis_tlast,
    input  wire        m_axis_tready,

    // ---- shared B-core boundary (one unified top owns the B-core) ----
    output wire        o_b_reset_request,
    output wire        o_b_start,
    output wire        o_b_pool_enable,
    output wire [`NPU_CFG_C_W-1:0] o_b_conv_cin,
    output wire [`NPU_CFG_C_W-1:0] o_b_conv_cout,
    output wire [`NPU_CFG_T_W-1:0] o_b_conv_out_len,
    output wire [`NPU_CFG_T_W-1:0] o_b_conv_post_out_len,
    output wire        o_b_act_valid,
    input  wire        i_b_act_ready,
    output wire signed [15:0] o_b_x0,
    output wire signed [15:0] o_b_x1,
    output wire signed [15:0] o_b_x2,
    output wire        o_b_weight_valid,
    input  wire        i_b_weight_ready,
    output wire [`NPU_WEIGHT_BUS_W-1:0] o_b_weight,
    output wire [`NPU_BIAS_BUS_W-1:0]   o_b_bias,
    output wire [`NPU_RSHIFT_BUS_W-1:0] o_b_rshift,
    input  wire [`NPU_GROUP_W-1:0] i_b_result_group_idx,
    input  wire        i_b_out_valid,
    output wire        o_b_out_ready,
    input  wire [`NPU_OUT_BUS_W-1:0] i_b_out_data,
    input  wire        i_b_busy,
    input  wire        i_b_done,
    input  wire [`NPU_CFG_C_W-1:0] i_b_conv_oc_group_idx,

    // Unified Conv1..9 parameter system. These ports are ignored when
    // USE_SHARED_PARAMS=0 so legacy standalone testbenches remain valid.
    input  wire i_shared_params_ready,
    input  wire [`NPU_WEIGHT_BUS_W-1:0] i_shared_weight,
    input  wire [`NPU_BIAS_BUS_W-1:0]   i_shared_bias,
    input  wire [`NPU_RSHIFT_BUS_W-1:0] i_shared_rshift,
    input  wire i_shared_weight_valid,
    input  wire i_shared_bias_valid,
    input  wire i_shared_rshift_valid,
    output wire o_b_rshift_valid
);

    // ============================================================
    // Conv1 layer constants (design doc S0) - fixed for this bitstream.
    // ============================================================
    localparam CONV1_CIN       = 1;
    localparam CONV1_COUT      = 64;
    localparam OC_GROUP_COUNT  = 4;

    localparam WEIGHT_BEATS_PER_GROUP = 24;  // 96B/group / 4B
    localparam WEIGHT_TOTAL_BEATS     = 96;  // 4 groups
    localparam BIAS_BEATS_PER_GROUP   = 32;  // 16 lanes * 2 beats/lane
    localparam BIAS_TOTAL_BEATS       = 128;
    localparam RSHIFT_BEATS_PER_GROUP = 4;   // 16B/group / 4B
    localparam RSHIFT_TOTAL_BEATS     = 16;

    // Derived from CONV1_OUT_LEN (P1 fix) - widths sized to the parameter,
    // not hardcoded for the full 106,667 case, so a reduced-T_out xsim
    // instance (e.g. 8 or 9) gets correspondingly small counters instead of
    // silently keeping 19-bit registers with unreachable upper bits.
    localparam OUTPUT_GROUP_CNT_W = $clog2(4 * CONV1_OUT_LEN);
    localparam [OUTPUT_GROUP_CNT_W-1:0] LAST_OUTPUT_GROUP = (4 * CONV1_OUT_LEN) - 1;

    // Total real activation halfwords for the whole Conv1 frame: t=0
    // consumes 2, every other of the (T_OUT-1) windows consumes 3
    // (design doc S0/S3): 2 + 3*(T_OUT-1) = 3*T_OUT - 1.
    localparam SAMPLE_CNT_W = $clog2(3 * CONV1_OUT_LEN);
    localparam [SAMPLE_CNT_W-1:0] RUN_EXPECTED_SAMPLES = (3 * CONV1_OUT_LEN) - 1;

    localparam [7:0] ERR_LOAD_TKEEP  = 8'h10;
    localparam [7:0] ERR_LOAD_TLAST  = 8'h11;
    localparam [7:0] ERR_LOAD_LENGTH = 8'h12;
    localparam [7:0] ERR_RUN_TKEEP   = 8'h13;
    localparam [7:0] ERR_RUN_TLAST   = 8'h14;
    localparam [7:0] ERR_RUN_LENGTH  = 8'h15;

    // ============================================================
    // SOFT_RESET -> registered one-cycle low pulse (same pattern as
    // npu_global_backend_adapter.v).
    // ============================================================
    reg b_rst_n_reg;

    // o_error_pulse is included alongside i_soft_reset_pulse (not just
    // reacted to via op_active/phase resets) because a RUN-phase error fires
    // AFTER B's own i_start has already been accepted - B's internal o_busy
    // only clears on its own natural loop completion (conv_compute_ctrl.v:
    // "if (!o_busy) begin if (i_start) ... end"), so without also resetting
    // b_rst_n_reg here, B would stay busy=1 forever and silently ignore
    // every future i_start pulse, hanging every subsequent operation.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            b_rst_n_reg <= 1'b0;
        else if (i_soft_reset_pulse || o_error_pulse)
            b_rst_n_reg <= 1'b0;
        else
            b_rst_n_reg <= 1'b1;
    end

    wire adapter_clear = !rst_n || !b_rst_n_reg;

    // ============================================================
    // op_active: true from an accepted START until wrapper-level o_done OR
    // o_error_pulse. o_busy is driven straight from op_active - unlike
    // GLOBAL, Conv1's LOAD sub-phase must already read BUSY=1 before B's own
    // busy ever goes high (design doc S8.1's "LOAD 서브페이즈 자체도 BUSY에
    // 포함"), and op_active already spans LOAD+RUN+DRAIN uniformly since it
    // only clears on the two terminal events.
    // ============================================================
    reg op_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_active <= 1'b0;
        end
        else if (adapter_clear) begin
            op_active <= 1'b0;
        end
        else if (i_start_pulse) begin
            op_active <= 1'b1;
        end
        else if (o_done || o_error_pulse) begin
            op_active <= 1'b0;
        end
    end

    assign o_busy = op_active;

    // ============================================================
    // LOAD phase state machine: weight -> bias -> rshift, strictly
    // sequential, each frame validated beat-by-beat.
    // ============================================================
    localparam [1:0] PHASE_LOAD_WEIGHT = 2'd0;
    localparam [1:0] PHASE_LOAD_BIAS   = 2'd1;
    localparam [1:0] PHASE_LOAD_RSHIFT = 2'd2;
    localparam [1:0] PHASE_RUN         = 2'd3;

    reg [1:0] phase;
    reg [6:0] beat_in_group;   // up to 31 (bias' 32 beats/group)
    reg [1:0] group_idx;       // 0..3
    reg [8:0] load_beat_cnt;   // 0..127 (bias' 128 total beats)

    reg [`NPU_WEIGHT_BUS_W-1:0] weight_local [0:OC_GROUP_COUNT-1];
    reg [`NPU_BIAS_BUS_W-1:0]   bias_local   [0:OC_GROUP_COUNT-1];
    reg [`NPU_RSHIFT_BUS_W-1:0] rshift_local [0:OC_GROUP_COUNT-1];

    wire is_load_phase = USE_SHARED_PARAMS ? 1'b0 : (phase != PHASE_RUN);

    wire [6:0] beats_per_group_m1 = (phase == PHASE_LOAD_WEIGHT) ? (WEIGHT_BEATS_PER_GROUP - 1) :
                                     (phase == PHASE_LOAD_BIAS)   ? (BIAS_BEATS_PER_GROUP - 1)   :
                                                                     (RSHIFT_BEATS_PER_GROUP - 1);
    wire [8:0] total_beats_m1     = (phase == PHASE_LOAD_WEIGHT) ? (WEIGHT_TOTAL_BEATS - 1) :
                                     (phase == PHASE_LOAD_BIAS)   ? (BIAS_TOTAL_BEATS - 1)   :
                                                                     (RSHIFT_TOTAL_BEATS - 1);

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

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase             <= PHASE_LOAD_WEIGHT;
            beat_in_group      <= 7'd0;
            group_idx          <= 2'd0;
            load_beat_cnt      <= 9'd0;
            b_start_pulse_reg <= 1'b0;
        end
        else if (adapter_clear || !op_active) begin
            phase             <= PHASE_LOAD_WEIGHT;
            beat_in_group      <= 7'd0;
            group_idx          <= 2'd0;
            load_beat_cnt      <= 9'd0;
            b_start_pulse_reg <= 1'b0;
        end
        else begin
            b_start_pulse_reg <= 1'b0;

            if (USE_SHARED_PARAMS && i_shared_params_ready) begin
                phase             <= PHASE_RUN;
                b_start_pulse_reg <= 1'b1;
            end
            else if (load_fire && !load_error) begin
                // Capture this beat into the phase-appropriate local buffer.
                case (phase)
                    PHASE_LOAD_WEIGHT: begin
                        weight_local[group_idx][beat_in_group*32 +: 32] <= s_axis_tdata;
                    end
                    PHASE_LOAD_BIAS: begin
                        // lane = beat_in_group>>1, low beat (even) = full
                        // 32b, high beat (odd) = only its lower 16b (upper
                        // 16b of the 64-bit DDR container truncated, design
                        // doc S2.2).
                        if (beat_in_group[0] == 1'b0)
                            bias_local[group_idx][(beat_in_group>>1)*48 +: 32] <= s_axis_tdata;
                        else
                            bias_local[group_idx][(beat_in_group>>1)*48 + 32 +: 16] <= s_axis_tdata[15:0];
                    end
                    PHASE_LOAD_RSHIFT: begin
                        rshift_local[group_idx][beat_in_group*32 +: 32] <= s_axis_tdata;
                    end
                    default: ;
                endcase

                load_beat_cnt <= load_beat_cnt + 1'b1;

                if (beat_in_group == beats_per_group_m1) begin
                    beat_in_group <= 7'd0;
                    group_idx      <= group_idx + 2'd1;
                end
                else begin
                    beat_in_group <= beat_in_group + 7'd1;
                end

                if (expected_last_beat) begin
                    beat_in_group <= 7'd0;
                    group_idx      <= 2'd0;
                    load_beat_cnt  <= 9'd0;
                    case (phase)
                        PHASE_LOAD_WEIGHT: phase <= PHASE_LOAD_BIAS;
                        PHASE_LOAD_BIAS:   phase <= PHASE_LOAD_RSHIFT;
                        PHASE_LOAD_RSHIFT: begin
                            phase             <= PHASE_RUN;
                            b_start_pulse_reg <= 1'b1;
                        end
                        default: ;
                    endcase
                end
            end
        end
    end

    // s_axis_tready: during LOAD, always ready (no backpressure needed - the
    // local buffers accept one beat per cycle unconditionally). During RUN,
    // gated by the halfword FIFO's fill level (below).
    wire run_hw_tready;
    assign s_axis_tready = op_active && b_rst_n_reg && !i_soft_reset_pulse &&
                            (USE_SHARED_PARAMS ? ((phase == PHASE_RUN) && run_hw_tready) :
                                                 (is_load_phase || run_hw_tready));

    // ============================================================
    // RUN phase: activation halfword FIFO + 3-tap window generation
    // (design doc S3). 4-deep FIFO of halfwords; a beat's low/high halfword
    // is pushed independently per TKEEP bit (RUN allows a final partial
    // beat, tkeep=4'h3, unlike LOAD).
    // ============================================================
    reg [15:0] hw_fifo [0:3];
    reg [2:0]  hw_count;
    reg [SAMPLE_CNT_W-1:0] run_sample_cnt;
    reg        run_input_complete;

    wire run_axis_fire  = s_axis_tvalid && s_axis_tready && (phase == PHASE_RUN);

    // ---- P2 fix: RUN frame validation, mirroring LOAD's TKEEP/TLAST/length
    // checks instead of unconditionally trusting the mask. Unlike LOAD, a
    // 4'h3 tkeep is legal here but ONLY on the beat that carries tlast - any
    // other combination (bad mask, tlast too early, or reaching the
    // expected sample count without tlast) is a malformed activation frame.
    // ----
    wire run_tkeep_valid_final    = (s_axis_tkeep == 4'hF) || (s_axis_tkeep == 4'h3);
    wire run_tkeep_valid_nonfinal = (s_axis_tkeep == 4'hF);
    wire run_tkeep_ok = s_axis_tlast ? run_tkeep_valid_final : run_tkeep_valid_nonfinal;

    wire [1:0] run_push_n_raw = {1'b0, s_axis_tkeep[1:0] == 2'b11} + {1'b0, s_axis_tkeep[3:2] == 2'b11};
    wire [SAMPLE_CNT_W-1:0] run_sample_cnt_next = run_sample_cnt + run_push_n_raw;
    wire run_expected_reached = (run_sample_cnt_next == RUN_EXPECTED_SAMPLES);

    wire run_tkeep_bad     = run_axis_fire && !run_tkeep_ok;
    wire run_tlast_early   = run_axis_fire && run_tkeep_ok && s_axis_tlast && !run_expected_reached;
    wire run_tlast_missing = run_axis_fire && run_tkeep_ok && run_expected_reached && !s_axis_tlast;
    wire run_error          = run_tkeep_bad || run_tlast_early || run_tlast_missing;

    wire [7:0] run_error_code = run_tkeep_bad     ? ERR_RUN_TKEEP  :
                                 run_tlast_early   ? ERR_RUN_TLAST  :
                                 run_tlast_missing ? ERR_RUN_LENGTH :
                                                      8'h00;

    // Combined LOAD/RUN error output - declared here (not immediately after
    // the LOAD section) because xvlog requires run_error/run_error_code's
    // wire declarations to textually precede this reference.
    assign o_error_pulse = op_active && ((USE_SHARED_PARAMS ? 1'b0 : load_error) || run_error);
    assign o_error_code  = is_load_phase ? load_error_code : run_error_code;

    // Only a validated (non-error) beat actually pushes into the halfword
    // FIFO or advances run_sample_cnt - a malformed beat aborts the whole
    // operation via o_error_pulse instead (op_active drops next cycle).
    wire run_push_low   = run_axis_fire && !run_error && s_axis_tkeep[1:0] == 2'b11;
    wire run_push_high  = run_axis_fire && !run_error && s_axis_tkeep[3:2] == 2'b11;
    wire [1:0] push_n   = {1'b0, run_push_low} + {1'b0, run_push_high};

    // P1 fix: the cycle a *validated* (non-error) final beat is accepted -
    // tlast, tkeep ok, and the expected sample count reached - close the
    // input side for good. Without this, s_axis_tready (via run_hw_tready)
    // would stay high as long as the FIFO has room, silently absorbing any
    // stray beat the driver sends after the legitimate last one instead of
    // flagging it. Held through DRAIN/DONE (only cleared when phase leaves
    // PHASE_RUN, i.e. the whole operation resets for the next START).
    wire run_input_final_fire = run_axis_fire && !run_error && s_axis_tlast && run_expected_reached;

    reg        first_window_consumed;
    reg        tap_valid;
    reg signed [15:0] tap_x0, tap_x1, tap_x2;

    wire [`NPU_CFG_C_W-1:0] conv_oc_group_idx_raw_w;  // driven from B core instance below, full CFG_C_W width
    wire [1:0] conv_oc_group_idx_w = conv_oc_group_idx_raw_w[1:0];  // Conv1 only ever uses OC groups 0-3
    wire       conv_step_fire;       // driven from B core instance below

    wire pop_threshold_met = !first_window_consumed ? (hw_count >= 3'd2) : (hw_count >= 3'd3);
    wire do_pop            = (phase == PHASE_RUN) && !tap_valid && pop_threshold_met;
    wire [1:0] pop_n        = !first_window_consumed ? 2'd2 : 2'd3;

    assign run_hw_tready = (hw_count < 3'd3) && !run_input_complete;

    integer sh_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hw_count            <= 3'd0;
            run_sample_cnt      <= {SAMPLE_CNT_W{1'b0}};
            run_input_complete  <= 1'b0;
        end
        else if (adapter_clear || (phase != PHASE_RUN)) begin
            hw_count            <= 3'd0;
            run_sample_cnt      <= {SAMPLE_CNT_W{1'b0}};
            run_input_complete  <= 1'b0;
        end
        else begin
            if (run_input_final_fire)
                run_input_complete <= 1'b1;

            if (run_axis_fire && !run_error)
                run_sample_cnt <= run_sample_cnt + push_n;

            // Shift out do_pop entries (from the head), then append the
            // newly pushed halfword(s) at the (post-shift) tail. Both can
            // happen the same cycle - push is only gated on hw_count<3 while
            // pop only fires at hw_count>=2, so the two windows do overlap
            // at hw_count==2, and this combined shift+append handles it.
            for (sh_i = 0; sh_i < 4; sh_i = sh_i + 1) begin
                if (do_pop) begin
                    if (sh_i + pop_n < 4)
                        hw_fifo[sh_i] <= hw_fifo[sh_i + pop_n];
                end
            end
            if (run_push_low)
                hw_fifo[(do_pop ? hw_count - pop_n : hw_count)] <= s_axis_tdata[15:0];
            if (run_push_high)
                hw_fifo[(do_pop ? hw_count - pop_n : hw_count) + {1'b0, run_push_low}] <= s_axis_tdata[31:16];

            hw_count <= hw_count + push_n - (do_pop ? pop_n : 2'd0);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            first_window_consumed <= 1'b0;
            tap_valid              <= 1'b0;
            tap_x0 <= 16'sd0; tap_x1 <= 16'sd0; tap_x2 <= 16'sd0;
        end
        else if (adapter_clear || (phase != PHASE_RUN)) begin
            first_window_consumed <= 1'b0;
            tap_valid              <= 1'b0;
        end
        else if (do_pop) begin
            if (!first_window_consumed) begin
                tap_x0 <= 16'sd0;          // left pad, design doc S0
                tap_x1 <= hw_fifo[0];
                tap_x2 <= hw_fifo[1];
                first_window_consumed <= 1'b1;
            end
            else begin
                tap_x0 <= hw_fifo[0];
                tap_x1 <= hw_fifo[1];
                tap_x2 <= hw_fifo[2];
            end
            tap_valid <= 1'b1;
        end
        else if (conv_step_fire && (conv_oc_group_idx_w == 2'd3)) begin
            tap_valid <= 1'b0;   // this time_idx's 4th (last) OC group step
                                  // just fired - next step needs a new window
        end
    end

    // ============================================================
    // Weight/bias/rshift combinational selection - issue-side for
    // weight/bias, result-side for rshift (frozen spec S5, design doc S2.3).
    // ============================================================
    wire [`NPU_GROUP_W-1:0] result_group_idx_w; // driven from B core instance below

    wire [`NPU_WEIGHT_BUS_W-1:0] sel_weight = USE_SHARED_PARAMS ? i_shared_weight : weight_local[conv_oc_group_idx_w];
    wire [`NPU_BIAS_BUS_W-1:0]   sel_bias   = USE_SHARED_PARAMS ? i_shared_bias : bias_local[conv_oc_group_idx_w];
    wire [`NPU_RSHIFT_BUS_W-1:0] sel_rshift = USE_SHARED_PARAMS ? i_shared_rshift : rshift_local[result_group_idx_w[1:0]];
    wire shared_issue_ready = !USE_SHARED_PARAMS || (i_shared_weight_valid && i_shared_bias_valid);
    wire tap_issue_valid = tap_valid && shared_issue_ready;

    // ============================================================
    // Output side: drain one 256-bit CONV1 result beat into 8 AXIS words.
    // Same DASM_IDLE/DASM_DRAIN pattern as npu_global_backend_adapter.v, but
    // the "is this the last group" decision uses this adapter's own
    // output_group_count (design doc S8.3), not a B result-side index.
    // ============================================================
    localparam DASM_IDLE  = 1'b0;
    localparam DASM_DRAIN = 1'b1;

    reg                       dasm_state;
    reg [`NPU_OUT_BUS_W-1:0]  dasm_shift_reg;
    reg [2:0]                 dasm_word_idx;
    reg                       dasm_is_last_group;
    reg [OUTPUT_GROUP_CNT_W-1:0] output_group_count;

    wire core_out_valid_w, core_out_ready_w;
    wire [`NPU_OUT_BUS_W-1:0] core_out_data_w;

    wire core_out_fire = core_out_valid_w && core_out_ready_w;
    wire m_axis_fire    = m_axis_tvalid && m_axis_tready;

    assign core_out_ready_w = (dasm_state == DASM_IDLE);
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
            output_group_count  <= {OUTPUT_GROUP_CNT_W{1'b0}};
        end
        else if (adapter_clear || !op_active) begin
            dasm_state          <= DASM_IDLE;
            dasm_shift_reg      <= {`NPU_OUT_BUS_W{1'b0}};
            dasm_word_idx       <= 3'd0;
            dasm_is_last_group  <= 1'b0;
            output_group_count  <= {OUTPUT_GROUP_CNT_W{1'b0}};
        end
        else begin
            case (dasm_state)
                DASM_IDLE: begin
                    if (core_out_fire) begin
                        dasm_shift_reg      <= core_out_data_w;
                        dasm_is_last_group  <= (output_group_count == LAST_OUTPUT_GROUP);
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
    // Shared B-core boundary.  This frontend retains Conv1's dedicated
    // windowing/LOAD/DRAIN FSM but no longer owns a private B-core instance.
    // ============================================================
    wire act_ready_w = i_b_act_ready;

    // Accepted-step condition from the adapter's own side: i_act_valid
    // (=tap_valid) paired with B's o_act_ready. Per the frozen operand
    // contract (o_act_ready depends on i_weight_valid, which we always tie
    // equal to tap_valid, so this single AND already reflects the paired
    // activation+weight acceptance for the current OC-group step.
    assign conv_step_fire = tap_issue_valid && act_ready_w;

    assign result_group_idx_w      = i_b_result_group_idx;
    assign conv_oc_group_idx_raw_w = i_b_conv_oc_group_idx;
    assign core_out_valid_w        = i_b_out_valid;
    assign core_out_data_w         = i_b_out_data;

    assign o_b_reset_request    = i_soft_reset_pulse || o_error_pulse;
    assign o_b_start            = b_start_pulse_reg;
    assign o_b_pool_enable      = 1'b0;
    assign o_b_conv_cin         = CONV1_CIN[`NPU_CFG_C_W-1:0];
    assign o_b_conv_cout        = CONV1_COUT[`NPU_CFG_C_W-1:0];
    assign o_b_conv_out_len     = CONV1_OUT_LEN[`NPU_CFG_T_W-1:0];
    assign o_b_conv_post_out_len= CONV1_OUT_LEN[`NPU_CFG_T_W-1:0];
    assign o_b_act_valid        = tap_issue_valid;
    assign o_b_x0               = tap_x0;
    assign o_b_x1               = tap_x1;
    assign o_b_x2               = tap_x2;
    assign o_b_weight_valid     = tap_issue_valid;
    assign o_b_weight           = sel_weight;
    assign o_b_bias             = sel_bias;
    assign o_b_rshift           = sel_rshift;
    assign o_b_rshift_valid     = USE_SHARED_PARAMS ? i_shared_rshift_valid : 1'b1;
    assign o_b_out_ready        = core_out_ready_w;

endmodule
