`timescale 1ns / 1ps

module b_compute_top_16 #(
    parameter DATA_W          = 16,
    parameter WEIGHT_W        = 16,
    parameter ACC_W           = 48,
    parameter OUT_W           = 16,
    parameter SHIFT_W         = 8,
    parameter CFG_C_W         = 10,
    parameter CFG_T_W         = 20,
    parameter CFG_DIM_W       = 10,
    parameter GROUP_W         = 6,
    parameter MAX_CONV_GROUPS = 16,
    parameter RSHIFT_VALID_GATING = 0,
    parameter GLOBAL_GROUP_W  = 4,
    parameter GLOBAL_TIME_W   = 5
)(
    input  wire clk,
    input  wire rst_n,

    // 00=CONV, 01=FC, 10=GLOBAL. 11 is reserved.
    input  wire [1:0] i_op_mode,
    input  wire       i_start,

    // CONV / FC sub-mode and configuration.
    input  wire                    i_pool_enable,
    input  wire                    i_fc1_mode,
    input  wire [CFG_C_W-1:0]      i_conv_cin,
    input  wire [CFG_C_W-1:0]      i_conv_cout,
    input  wire [CFG_T_W-1:0]      i_conv_out_len,
    // Precomputed post-pool length - see unified_compute_core_16.v's port
    // comment (Docs/C_to_B_ConvOutLen_ConfigTiming_Review_Request_v1.md).
    input  wire [CFG_T_W-1:0]      i_conv_post_out_len,
    input  wire [CFG_DIM_W-1:0]    i_fc_in_dim,
    input  wire [CFG_DIM_W-1:0]    i_fc_out_dim,
    input  wire [CFG_DIM_W-1:0]    i_fc_chunk_count,
    input  wire [1:0]              i_fc_last_valid_taps,

    // Shared CONV / FC operand interface.
    input  wire                     i_act_valid,
    output wire                     o_act_ready,
    input  wire [4*3*DATA_W-1:0]     i_x,

    input  wire                      i_weight_valid,
    output wire                      o_weight_ready,
    input  wire [16*4*3*WEIGHT_W-1:0] i_weight,
    input  wire [16*ACC_W-1:0]       i_bias,

    // Result-aligned per-OC shift for CONV / FC.
    input  wire [16*SHIFT_W-1:0]     i_rshift,
    input  wire                       i_rshift_valid,
    output wire [GROUP_W-1:0]        o_result_group_idx,
    output wire [CFG_T_W-1:0]        o_result_time_idx,

    // GLOBAL input interface.
    // The B-side order is fixed as Time outer / OC-group inner.
    // The wrapper generates group / first / last metadata internally.
    input  wire                       i_global_valid,
    output wire                       o_global_ready,
    input  wire [16*DATA_W-1:0]       i_global_data,
    output wire [GLOBAL_GROUP_W-1:0]  o_global_group_idx,
    output wire [GLOBAL_TIME_W-1:0]   o_global_time_idx,

    // Unified result interface.
    output wire [1:0]                 o_output_op_mode,
    output wire                       o_valid,
    input  wire                       i_out_ready,
    output wire [16*OUT_W-1:0]        o_data,
    output wire [15:0]                o_lane_valid_mask,
    output wire [GROUP_W-1:0]         o_group_idx,

    output wire                       o_busy,
    output wire                       o_done,

    // CONV / FC issue-side addressing/debug from the unified core.
    output wire [CFG_C_W-1:0]         o_conv_cin_idx,
    output wire [CFG_C_W-1:0]         o_conv_oc_group_idx,
    output wire [CFG_T_W-1:0]         o_conv_time_idx,
    output wire [CFG_DIM_W-1:0]       o_fc_chunk_idx,
    output wire [CFG_DIM_W-1:0]       o_fc_out_group_idx
);

    localparam [1:0] OP_CONV   = 2'b00;
    localparam [1:0] OP_FC     = 2'b01;
    localparam [1:0] OP_GLOBAL = 2'b10;

    localparam [GLOBAL_GROUP_W-1:0] GLOBAL_LAST_GROUP = 4'd15;
    localparam [GLOBAL_TIME_W-1:0]  GLOBAL_LAST_TIME  = 5'd16;

    reg [1:0] op_mode_reg;
    reg       global_active_reg;
    reg       global_done_reg;
    reg       global_input_complete_reg;

    wire unified_busy;
    wire unified_done;

    wire start_accept;
    wire start_unified;
    wire start_global;

    assign start_accept =
        i_start
        && !unified_busy
        && !global_active_reg
        && (i_op_mode != 2'b11);

    assign start_unified =
        start_accept
        && ((i_op_mode == OP_CONV) || (i_op_mode == OP_FC));

    assign start_global =
        start_accept
        && (i_op_mode == OP_GLOBAL);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_mode_reg <= OP_CONV;
        end
        else if (start_accept) begin
            op_mode_reg <= i_op_mode;
        end
    end

    // ============================================================
    // Existing Unified CONV / FC Core
    // ============================================================
    wire                     unified_act_ready;
    wire                     unified_weight_ready;
    wire [GROUP_W-1:0]       unified_result_group_idx;
    wire [CFG_T_W-1:0]       unified_result_time_idx;
    wire                     unified_valid;
    wire [16*OUT_W-1:0]      unified_data;
    wire [15:0]              unified_lane_valid_mask;
    wire [GROUP_W-1:0]       unified_group_idx;

    wire unified_out_ready;

    // Unified and GLOBAL operations are mutually exclusive by start_accept,
    // and the unified core accepts data only while its own controller is
    // busy.  Mode-gating these handshake signals is therefore redundant and
    // creates a long op_mode_reg -> controller-CE cone across the core.
    assign unified_out_ready = i_out_ready;

    unified_compute_core_16 #(
        .DATA_W          (DATA_W),
        .WEIGHT_W        (WEIGHT_W),
        .ACC_W           (ACC_W),
        .OUT_W           (OUT_W),
        .SHIFT_W         (SHIFT_W),
        .CFG_C_W         (CFG_C_W),
        .CFG_T_W         (CFG_T_W),
        .CFG_DIM_W       (CFG_DIM_W),
        .GROUP_W         (GROUP_W),
        .MAX_CONV_GROUPS (MAX_CONV_GROUPS),
        .RSHIFT_VALID_GATING (RSHIFT_VALID_GATING)
    ) u_unified_compute_core_16 (
        .clk                 (clk),
        .rst_n               (rst_n),

        .i_mode              (i_op_mode == OP_FC),
        .i_start             (start_unified),
        .i_pool_enable       (i_pool_enable),
        .i_fc1_mode          (i_fc1_mode),

        .i_conv_cin          (i_conv_cin),
        .i_conv_cout         (i_conv_cout),
        .i_conv_out_len      (i_conv_out_len),
        .i_conv_post_out_len (i_conv_post_out_len),
        .i_fc_in_dim         (i_fc_in_dim),
        .i_fc_out_dim        (i_fc_out_dim),
        .i_fc_chunk_count    (i_fc_chunk_count),
        .i_fc_last_valid_taps(i_fc_last_valid_taps),

        .i_act_valid         (i_act_valid),
        .o_act_ready         (unified_act_ready),
        .i_x                 (i_x),

        .i_weight_valid      (i_weight_valid),
        .o_weight_ready      (unified_weight_ready),
        .i_weight            (i_weight),
        .i_bias              (i_bias),

        .i_rshift            (i_rshift),
        .i_rshift_valid      (i_rshift_valid),
        .o_result_group_idx  (unified_result_group_idx),
        .o_result_time_idx   (unified_result_time_idx),

        .o_valid             (unified_valid),
        .i_out_ready         (unified_out_ready),
        .o_data              (unified_data),
        .o_lane_valid_mask   (unified_lane_valid_mask),
        .o_group_idx         (unified_group_idx),

        .o_busy              (unified_busy),
        .o_done              (unified_done),

        .o_conv_cin_idx      (o_conv_cin_idx),
        .o_conv_oc_group_idx (o_conv_oc_group_idx),
        .o_conv_time_idx     (o_conv_time_idx),
        .o_fc_chunk_idx      (o_fc_chunk_idx),
        .o_fc_out_group_idx  (o_fc_out_group_idx)
    );

    assign o_act_ready =
        (op_mode_reg == OP_GLOBAL)
        ? 1'b0
        : unified_act_ready;

    assign o_weight_ready =
        (op_mode_reg == OP_GLOBAL)
        ? 1'b0
        : unified_weight_ready;

    // Result metadata is consumed only by the selected CONV/FC frontend.
    // During GLOBAL those frontends are inactive, so forcing these buses to
    // zero through op_mode_reg is functionally redundant.  The zero mux also
    // formed a full combinational feedback path through ConvN rshift-cache
    // validity and postprocess ready back to the shared MAC stage-0 CEs.
    // Export the unified core's already-registered metadata directly.
    assign o_result_group_idx = unified_result_group_idx;
    assign o_result_time_idx  = unified_result_time_idx;

    // ============================================================
    // GLOBAL input metadata tracker
    // Frozen Conv9 Global input: 256 channels x 17 time samples.
    // One beat carries 16 channels, so 16 groups x 17 times.
    // ============================================================
    reg [GLOBAL_GROUP_W-1:0] global_group_idx_reg;
    reg [GLOBAL_TIME_W-1:0]  global_time_idx_reg;

    wire global_pool_ready;
    wire global_input_valid;
    wire global_input_fire;

    assign o_global_group_idx = global_group_idx_reg;
    assign o_global_time_idx  = global_time_idx_reg;

    assign global_input_valid =
        i_global_valid
        && global_active_reg
        && (op_mode_reg == OP_GLOBAL)
        && !global_input_complete_reg;

    assign o_global_ready =
        global_pool_ready
        && global_active_reg
        && (op_mode_reg == OP_GLOBAL)
        && !global_input_complete_reg;

    assign global_input_fire =
        global_input_valid
        && global_pool_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_group_idx_reg <= {GLOBAL_GROUP_W{1'b0}};
            global_time_idx_reg  <= {GLOBAL_TIME_W{1'b0}};
            global_input_complete_reg <= 1'b0;
        end
        else begin
            if (start_global) begin
                global_group_idx_reg <= {GLOBAL_GROUP_W{1'b0}};
                global_time_idx_reg  <= {GLOBAL_TIME_W{1'b0}};
                global_input_complete_reg <= 1'b0;
            end
            else if (global_input_fire) begin
                if ((global_group_idx_reg == GLOBAL_LAST_GROUP) &&
                    (global_time_idx_reg == GLOBAL_LAST_TIME)) begin
                    global_input_complete_reg <= 1'b1;
                end

                if (global_group_idx_reg == GLOBAL_LAST_GROUP) begin
                    global_group_idx_reg <= {GLOBAL_GROUP_W{1'b0}};

                    if (global_time_idx_reg != GLOBAL_LAST_TIME) begin
                        global_time_idx_reg <= global_time_idx_reg + 1'b1;
                    end
                end
                else begin
                    global_group_idx_reg <= global_group_idx_reg + 1'b1;
                end
            end
        end
    end

    // ============================================================
    // Existing bit-exact Global Pool
    // ============================================================
    wire                         global_valid;
    wire [16*DATA_W-1:0]         global_data;
    wire [15:0]                  global_lane_valid_mask;
    wire [GLOBAL_GROUP_W-1:0]    global_output_group_idx;
    wire                         global_out_ready;

    assign global_out_ready =
        i_out_ready
        && global_active_reg
        && (op_mode_reg == OP_GLOBAL);

    global_pool_16 #(
        .DATA_W     (DATA_W),
        .GROUP_W    (GLOBAL_GROUP_W),
        .MAX_GROUPS (16),
        .SUM_W      (20)
    ) u_global_pool_16 (
        .clk                 (clk),
        .rst_n               (rst_n),

        .i_valid             (global_input_valid),
        .o_ready             (global_pool_ready),
        .i_data              (i_global_data),
        .i_lane_valid_mask   (16'hFFFF),
        .i_group_idx         (global_group_idx_reg),
        .i_first_time        (global_time_idx_reg == {GLOBAL_TIME_W{1'b0}}),
        .i_last_time         (global_time_idx_reg == GLOBAL_LAST_TIME),

        .o_valid             (global_valid),
        .i_out_ready         (global_out_ready),
        .o_data              (global_data),
        .o_lane_valid_mask   (global_lane_valid_mask),
        .o_group_idx         (global_output_group_idx)
    );

    wire global_output_fire;
    wire global_final_output_fire;

    assign global_output_fire =
        global_valid
        && global_out_ready;

    assign global_final_output_fire =
        global_output_fire
        && (global_output_group_idx == GLOBAL_LAST_GROUP);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_active_reg <= 1'b0;
            global_done_reg   <= 1'b0;
        end
        else begin
            global_done_reg <= 1'b0;

            if (start_global) begin
                global_active_reg <= 1'b1;
            end

            if (global_final_output_fire) begin
                global_active_reg <= 1'b0;
                global_done_reg   <= 1'b1;
            end
        end
    end

    // ============================================================
    // Unified result routing
    // ============================================================
    assign o_output_op_mode = op_mode_reg;

    assign o_valid =
        (op_mode_reg == OP_GLOBAL)
        ? global_valid
        : unified_valid;

    assign o_data =
        (op_mode_reg == OP_GLOBAL)
        ? global_data
        : unified_data;

    assign o_lane_valid_mask =
        (op_mode_reg == OP_GLOBAL)
        ? global_lane_valid_mask
        : unified_lane_valid_mask;

    assign o_group_idx =
        (op_mode_reg == OP_GLOBAL)
        ? {{(GROUP_W-GLOBAL_GROUP_W){1'b0}}, global_output_group_idx}
        : unified_group_idx;

    assign o_busy =
        (op_mode_reg == OP_GLOBAL)
        ? global_active_reg
        : unified_busy;

    assign o_done =
        (op_mode_reg == OP_GLOBAL)
        ? global_done_reg
        : unified_done;

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && start_accept) begin
            if (i_op_mode == 2'b11) begin
                $display("[ERROR][B_TOP] Reserved op mode 2'b11 requested");
            end
        end
    end
`endif

endmodule
