`timescale 1ns / 1ps

module unified_compute_core_16 #(
    parameter DATA_W          = 16,
    parameter WEIGHT_W        = 16,
    parameter ACC_W           = 48,
    parameter OUT_W           = 16,
    parameter SHIFT_W         = 8,

    parameter CFG_C_W         = 10,
    parameter CFG_T_W         = 20,
    parameter CFG_DIM_W       = 10,

    // Must cover FC2 ceil(527/16)=33 groups -> indices 0..32.
    parameter GROUP_W         = 6,
    parameter MAX_CONV_GROUPS = 16,
    parameter RSHIFT_VALID_GATING = 0
)(
    input  wire clk,
    input  wire rst_n,

    // ============================================================
    // Layer Control
    // ============================================================
    // 0 = CONV, 1 = FC
    input  wire                    i_mode,
    input  wire                    i_start,

    // CONV sub-mode
    // 0 = Conv1  : ReLU -> Requant -> Sat
    // 1 = Conv2~9: ReLU -> MaxPool -> Requant -> Sat
    input  wire                    i_pool_enable,

    // FC sub-mode
    // 1 = FC1: ReLU -> Requant -> Sat
    // 0 = FC2:        Requant -> Sat
    input  wire                    i_fc1_mode,

    // ============================================================
    // CONV Configuration
    // ============================================================
    input  wire [CFG_C_W-1:0]      i_conv_cin,
    input  wire [CFG_C_W-1:0]      i_conv_cout,
    input  wire [CFG_T_W-1:0]      i_conv_out_len,
    // Post-pool output length, precomputed and registered by C before this
    // START edge (Docs/C_to_B_ConvOutLen_ConfigTiming_Review_Request_v1.md /
    // B_to_C_..._Response_v1.md): Conv1 -> i_conv_out_len passthrough,
    // Conv2~9 -> (i_conv_out_len+2)/3, FC/GLOBAL -> tie 0 (unused). B does
    // NOT perform any arithmetic on this value at START - it is latched
    // as-is, replacing the (i_conv_out_len+2)/3 division that used to run
    // combinationally on this same edge and violated 100MHz timing.
    input  wire [CFG_T_W-1:0]      i_conv_post_out_len,

    // ============================================================
    // FC Configuration
    // ============================================================
    input  wire [CFG_DIM_W-1:0]    i_fc_in_dim,
    input  wire [CFG_DIM_W-1:0]    i_fc_out_dim,
    input  wire [CFG_DIM_W-1:0]    i_fc_chunk_count,
    input  wire [1:0]              i_fc_last_valid_taps,

    // ============================================================
    // Shared Activation Input
    // ============================================================
    input  wire                     i_act_valid,
    output wire                     o_act_ready,

    input  wire [4*3*DATA_W-1:0]     i_x,

    // ============================================================
    // Shared Weight Input: 16 lanes x 3 taps
    // ============================================================
    input  wire                      i_weight_valid,
    output wire                      o_weight_ready,
    input  wire [16*4*3*WEIGHT_W-1:0] i_weight,

    // ============================================================
    // Folded Bias: 16 lanes x signed48 effective Bias
    // ============================================================
    input  wire [16*ACC_W-1:0]       i_bias,

    // ============================================================
    // Result-Aligned Per-OC Rshift
    //
    // i_rshift must correspond to o_result_group_idx.
    // ============================================================
    input  wire [16*SHIFT_W-1:0]     i_rshift,
    input  wire                       i_rshift_valid,
    output wire [GROUP_W-1:0]        o_result_group_idx,
    output wire [CFG_T_W-1:0]        o_result_time_idx,

    // ============================================================
    // Unified INT16 Output
    // ============================================================
    output wire                      o_valid,
    input  wire                      i_out_ready,

    output wire [16*OUT_W-1:0]       o_data,
    output wire [15:0]               o_lane_valid_mask,
    output wire [GROUP_W-1:0]        o_group_idx,

    // ============================================================
    // Integrated Status
    //
    // o_busy stays high until the final postprocessed output beat
    // is accepted downstream.
    // o_done pulses for one cycle on that final output handshake.
    // ============================================================
    output wire                      o_busy,
    output reg                       o_done,

    // ============================================================
    // Issue-Side Debug / Addressing
    // ============================================================
    output wire [CFG_C_W-1:0]        o_conv_cin_idx,
    output wire [CFG_C_W-1:0]        o_conv_oc_group_idx,
    output wire [CFG_T_W-1:0]        o_conv_time_idx,

    output wire [CFG_DIM_W-1:0]      o_fc_chunk_idx,
    output wire [CFG_DIM_W-1:0]      o_fc_out_group_idx
);

    localparam MODE_CONV = 1'b0;
    localparam MODE_FC   = 1'b1;


    // ============================================================
    // Integrated Layer State / Latched Configuration
    // ============================================================
    reg active_reg;
    reg mode_reg;
    // Output selection is physically distributed per 16-bit lane.  Driving
    // the full 256-bit result mux from mode_reg made one control net span the
    // entire result datapath.  These START-latched copies preserve the same
    // operation-stable mode without adding an output cycle.
    (* keep = "true" *) reg [15:0] output_mode_lane_reg;
    (* keep = "true" *) reg        output_mode_meta_reg;
    reg pool_enable_reg;
    reg fc1_mode_reg;

    reg [CFG_C_W-1:0]   conv_cout_cfg_reg;
    reg [CFG_T_W-1:0]   conv_out_len_cfg_reg;
    reg [CFG_T_W-1:0]   conv_post_out_len_reg;
    reg [CFG_DIM_W-1:0] fc_out_dim_cfg_reg;

    assign o_busy = active_reg;

    wire start_accept;

    assign start_accept =
        i_start
        &&
        !active_reg;


    // ============================================================
    // CONV Final Indices
    // ============================================================
    wire [CFG_C_W-1:0] conv_last_group_wide;
    wire [GROUP_W-1:0] conv_last_group_idx;
    wire [CFG_T_W-1:0] conv_last_time_idx;
    wire [CFG_T_W-1:0] conv_last_post_time_idx;

    assign conv_last_group_wide =
        (conv_cout_cfg_reg >> 4)
        -
        1'b1;

    assign conv_last_group_idx =
        conv_last_group_wide[GROUP_W-1:0];

    assign conv_last_time_idx =
        conv_out_len_cfg_reg
        -
        1'b1;

    assign conv_last_post_time_idx =
        conv_post_out_len_reg
        -
        1'b1;


    // ============================================================
    // FC Final Group
    // ============================================================
    wire [CFG_DIM_W-1:0] fc_group_count_wide;
    wire [CFG_DIM_W-1:0] fc_last_group_wide;
    wire [GROUP_W-1:0]   fc_last_group_idx;

    assign fc_group_count_wide =
        (fc_out_dim_cfg_reg + 15)
        >>
        4;

    assign fc_last_group_wide =
        fc_group_count_wide
        -
        1'b1;

    assign fc_last_group_idx =
        fc_last_group_wide[GROUP_W-1:0];


    // ============================================================
    // One Shared Bias-Aware 48-MAC Core
    // ============================================================
    wire                    core_result_valid;
    wire                    core_result_ready;
    wire [16*ACC_W-1:0]     core_result;
    wire [15:0]             core_lane_valid_mask;

    wire core_busy_unused;
    wire core_done_unused;

    shared_mac_acc_core #(
        .DATA_W      (DATA_W),
        .WEIGHT_W    (WEIGHT_W),
        .ACC_W       (ACC_W),
        .CFG_C_W     (CFG_C_W),
        .CFG_T_W     (CFG_T_W),
        .CFG_DIM_W   (CFG_DIM_W)
    ) u_shared_mac_acc_core (
        .clk                 (clk),
        .rst_n               (rst_n),

        .i_mode              (i_mode),
        .i_start             (start_accept),

        .i_conv_cin          (i_conv_cin),
        .i_conv_cout         (i_conv_cout),
        .i_conv_out_len      (i_conv_out_len),

        .i_fc_in_dim         (i_fc_in_dim),
        .i_fc_out_dim        (i_fc_out_dim),
        .i_fc_chunk_count    (i_fc_chunk_count),
        .i_fc_last_valid_taps(i_fc_last_valid_taps),

        .i_act_valid         (i_act_valid),
        .o_act_ready         (o_act_ready),

        .i_x                 (i_x),

        .i_weight_valid      (i_weight_valid),
        .o_weight_ready      (o_weight_ready),
        .i_weight            (i_weight),

        .i_bias              (i_bias),

        .i_result_ready      (core_result_ready),
        .o_result_valid      (core_result_valid),
        .o_result            (core_result),
        .o_lane_valid_mask   (core_lane_valid_mask),

        .o_busy              (core_busy_unused),
        .o_done              (core_done_unused),

        .o_conv_cin_idx      (o_conv_cin_idx),
        .o_conv_oc_group_idx (o_conv_oc_group_idx),
        .o_conv_time_idx     (o_conv_time_idx),

        .o_fc_chunk_idx      (o_fc_chunk_idx),
        .o_fc_out_group_idx  (o_fc_out_group_idx)
    );


    // ============================================================
    // Result-Aligned CONV Metadata Tracker
    // ============================================================
    reg [GROUP_W-1:0] conv_result_group_idx_reg;
    reg [CFG_T_W-1:0] conv_result_time_idx_reg;
    reg [1:0]         conv_result_time_mod3_reg;

    wire conv_result_first_time;
    wire conv_result_last_time;

    assign conv_result_first_time =
        (conv_result_time_idx_reg == {CFG_T_W{1'b0}});

    assign conv_result_last_time =
        (conv_result_time_idx_reg == conv_last_time_idx);


    // ============================================================
    // Result-Aligned FC Metadata Tracker
    // ============================================================
    reg [GROUP_W-1:0] fc_result_group_idx_reg;


    // ============================================================
    // Shared-Core Result Hold Buffer
    //
    // This one-entry elastic stage freezes the MAC result, rshift and
    // result-aligned metadata before either postprocess pipeline consumes
    // them.  Only the valid bit is reset; payload bits are don't-care while
    // invalid, avoiding a wide reset tree on the 900+ payload registers.
    // ============================================================
    reg                         result_hold_valid_reg;
    reg [16*ACC_W-1:0]          result_hold_acc_reg;
    reg [15:0]                  result_hold_lane_mask_reg;
    reg [16*SHIFT_W-1:0]        result_hold_rshift_reg;
    reg                         result_hold_mode_reg;
    reg                         result_hold_pool_enable_reg;
    reg                         result_hold_fc1_mode_reg;
    reg [GROUP_W-1:0]           result_hold_group_idx_reg;
    reg [CFG_T_W-1:0]           result_hold_time_idx_reg;
    reg [1:0]                   result_hold_time_mod3_reg;
    reg                         result_hold_first_time_reg;
    reg                         result_hold_last_time_reg;

    wire result_hold_ready;


    // ============================================================
    // Unified Rshift Address Metadata
    // ============================================================
    assign o_result_group_idx =
        (mode_reg == MODE_FC)
        ?
        fc_result_group_idx_reg
        :
        conv_result_group_idx_reg;

    assign o_result_time_idx =
        (mode_reg == MODE_CONV)
        ?
        conv_result_time_idx_reg
        :
        {CFG_T_W{1'b0}};


    // ============================================================
    // CONV Postprocess
    // ============================================================
    wire conv_post_ready;
    wire conv_post_valid;
    wire [16*OUT_W-1:0] conv_post_data;
    wire [15:0]         conv_post_lane_mask;
    wire [GROUP_W-1:0] conv_post_group_idx;

    wire conv_post_in_valid;
    wire conv_post_out_ready;

    assign conv_post_in_valid =
        result_hold_valid_reg
        &&
        (result_hold_mode_reg == MODE_CONV);

    assign conv_post_out_ready =
        i_out_ready
        &&
        active_reg
        &&
        (mode_reg == MODE_CONV);

    conv_postprocess_16 #(
        .ACC_W      (ACC_W),
        .OUT_W      (OUT_W),
        .SHIFT_W    (SHIFT_W),
        .GROUP_W    (GROUP_W),
        .MAX_GROUPS (MAX_CONV_GROUPS)
    ) u_conv_postprocess_16 (
        .clk                 (clk),
        .rst_n               (rst_n),

        .i_valid             (conv_post_in_valid),
        .o_ready             (conv_post_ready),

        .i_acc               (result_hold_acc_reg),
        .i_rshift            (result_hold_rshift_reg),
        .i_lane_valid_mask   (result_hold_lane_mask_reg),
        .i_group_idx         (result_hold_group_idx_reg),

        .i_time_mod3         (result_hold_time_mod3_reg),
        .i_first_time        (result_hold_first_time_reg),
        .i_last_time         (result_hold_last_time_reg),

        .i_pool_enable       (result_hold_pool_enable_reg),

        .o_valid             (conv_post_valid),
        .i_out_ready         (conv_post_out_ready),

        .o_data              (conv_post_data),
        .o_lane_valid_mask   (conv_post_lane_mask),
        .o_group_idx         (conv_post_group_idx)
    );


    // ============================================================
    // FC Postprocess
    // ============================================================
    wire fc_post_ready;
    wire fc_post_valid;
    wire [16*OUT_W-1:0] fc_post_data;
    wire [15:0]         fc_post_lane_mask;
    wire [GROUP_W-1:0] fc_post_group_idx;

    wire fc_post_in_valid;
    wire fc_post_out_ready;

    assign fc_post_in_valid =
        result_hold_valid_reg
        &&
        (result_hold_mode_reg == MODE_FC);

    assign fc_post_out_ready =
        i_out_ready
        &&
        active_reg
        &&
        (mode_reg == MODE_FC);

    fc_postprocess_16 #(
        .ACC_W   (ACC_W),
        .OUT_W   (OUT_W),
        .SHIFT_W (SHIFT_W),
        .GROUP_W (GROUP_W)
    ) u_fc_postprocess_16 (
        .clk                 (clk),
        .rst_n               (rst_n),

        .i_valid             (fc_post_in_valid),
        .o_ready             (fc_post_ready),

        .i_acc               (result_hold_acc_reg),
        .i_rshift            (result_hold_rshift_reg),
        .i_lane_valid_mask   (result_hold_lane_mask_reg),
        .i_group_idx         (result_hold_group_idx_reg),
        .i_fc1_mode          (result_hold_fc1_mode_reg),

        .o_valid             (fc_post_valid),
        .i_out_ready         (fc_post_out_ready),

        .o_data              (fc_post_data),
        .o_lane_valid_mask   (fc_post_lane_mask),
        .o_group_idx         (fc_post_group_idx)
    );


    // ============================================================
    // Shared Core Backpressure Routing
    // ============================================================
    wire rshift_ready = !RSHIFT_VALID_GATING || i_rshift_valid;

    assign result_hold_ready =
        !result_hold_valid_reg
        ||
        ((result_hold_mode_reg == MODE_FC)
         ? fc_post_ready
         : conv_post_ready);

    assign core_result_ready =
        rshift_ready
        &&
        result_hold_ready;

    wire core_result_fire;

    assign core_result_fire =
        core_result_valid
        &&
        core_result_ready;


    // Valid state follows the conventional one-entry elastic-buffer rule:
    // retain while stalled, consume when downstream is ready, and replace
    // in the same cycle when a new shared-core result is accepted.
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin
            result_hold_valid_reg <= 1'b0;
        end
        else if (start_accept) begin
            result_hold_valid_reg <= 1'b0;
        end
        else if (result_hold_ready) begin
            result_hold_valid_reg <= core_result_fire;
        end

    end


    // Payload has no functional value while result_hold_valid_reg is low,
    // so it intentionally has no reset.  This prevents the buffer from
    // introducing a wide reset-fanout timing cone.
    always @(posedge clk) begin

        if (core_result_fire) begin
            result_hold_acc_reg          <= core_result;
            result_hold_lane_mask_reg    <= core_lane_valid_mask;
            result_hold_rshift_reg       <= i_rshift;
            result_hold_mode_reg         <= mode_reg;
            result_hold_pool_enable_reg  <= pool_enable_reg;
            result_hold_fc1_mode_reg     <= fc1_mode_reg;
            result_hold_time_idx_reg     <= conv_result_time_idx_reg;
            result_hold_time_mod3_reg    <= conv_result_time_mod3_reg;
            result_hold_first_time_reg   <= conv_result_first_time;
            result_hold_last_time_reg    <= conv_result_last_time;

            if (mode_reg == MODE_FC) begin
                result_hold_group_idx_reg <= fc_result_group_idx_reg;
            end
            else begin
                result_hold_group_idx_reg <= conv_result_group_idx_reg;
            end
        end

    end


    // ============================================================
    // Unified Output Routing
    // ============================================================
    assign o_valid =
        (output_mode_meta_reg == MODE_FC)
        ?
        fc_post_valid
        :
        conv_post_valid;

    genvar output_lane;
    generate
        for (output_lane = 0; output_lane < 16; output_lane = output_lane + 1) begin : GEN_OUTPUT_MODE_LANE
            assign o_data[output_lane*OUT_W +: OUT_W] =
                (output_mode_lane_reg[output_lane] == MODE_FC)
                ? fc_post_data[output_lane*OUT_W +: OUT_W]
                : conv_post_data[output_lane*OUT_W +: OUT_W];
        end
    endgenerate

    assign o_lane_valid_mask =
        (output_mode_meta_reg == MODE_FC)
        ?
        fc_post_lane_mask
        :
        conv_post_lane_mask;

    assign o_group_idx =
        (output_mode_meta_reg == MODE_FC)
        ?
        fc_post_group_idx
        :
        conv_post_group_idx;


    // ============================================================
    // Result Metadata Advance
    // ============================================================
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            conv_result_group_idx_reg <=
                {GROUP_W{1'b0}};

            conv_result_time_idx_reg <=
                {CFG_T_W{1'b0}};

            conv_result_time_mod3_reg <=
                2'd0;

            fc_result_group_idx_reg <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (start_accept) begin

                conv_result_group_idx_reg <=
                    {GROUP_W{1'b0}};

                conv_result_time_idx_reg <=
                    {CFG_T_W{1'b0}};

                conv_result_time_mod3_reg <=
                    2'd0;

                fc_result_group_idx_reg <=
                    {GROUP_W{1'b0}};

            end
            else if (core_result_fire) begin

                if (mode_reg == MODE_CONV) begin

                    if (
                        conv_result_group_idx_reg
                        ==
                        conv_last_group_idx
                    ) begin

                        if (
                            conv_result_time_idx_reg
                            !=
                            conv_last_time_idx
                        ) begin

                            conv_result_group_idx_reg <=
                                {GROUP_W{1'b0}};

                            conv_result_time_idx_reg <=
                                conv_result_time_idx_reg
                                +
                                1'b1;

                            if (
                                conv_result_time_mod3_reg
                                ==
                                2'd2
                            ) begin

                                conv_result_time_mod3_reg <=
                                    2'd0;

                            end
                            else begin

                                conv_result_time_mod3_reg <=
                                    conv_result_time_mod3_reg
                                    +
                                    1'b1;

                            end

                        end

                    end
                    else begin

                        conv_result_group_idx_reg <=
                            conv_result_group_idx_reg
                            +
                            1'b1;

                    end

                end
                else begin

                    if (
                        fc_result_group_idx_reg
                        !=
                        fc_last_group_idx
                    ) begin

                        fc_result_group_idx_reg <=
                            fc_result_group_idx_reg
                            +
                            1'b1;

                    end

                end

            end

        end

    end


    // ============================================================
    // CONV Postprocessed Output Time Tracker
    // ============================================================
    reg [CFG_T_W-1:0] conv_output_time_idx_reg;

    wire conv_output_fire;

    assign conv_output_fire =
        conv_post_valid
        &&
        conv_post_out_ready;

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            conv_output_time_idx_reg <=
                {CFG_T_W{1'b0}};

        end
        else begin

            if (start_accept) begin

                conv_output_time_idx_reg <=
                    {CFG_T_W{1'b0}};

            end
            else if (
                (mode_reg == MODE_CONV)
                &&
                conv_output_fire
                &&
                (conv_post_group_idx == conv_last_group_idx)
            ) begin

                if (
                    conv_output_time_idx_reg
                    !=
                    conv_last_post_time_idx
                ) begin

                    conv_output_time_idx_reg <=
                        conv_output_time_idx_reg
                        +
                        1'b1;

                end

            end

        end

    end


    // ============================================================
    // Final Output Completion
    // ============================================================
    wire fc_output_fire;
    wire conv_final_output_fire;
    wire fc_final_output_fire;
    wire selected_final_output_fire;

    assign fc_output_fire =
        fc_post_valid
        &&
        fc_post_out_ready;

    assign conv_final_output_fire =
        conv_output_fire
        &&
        (conv_post_group_idx == conv_last_group_idx)
        &&
        (conv_output_time_idx_reg == conv_last_post_time_idx);

    assign fc_final_output_fire =
        fc_output_fire
        &&
        (fc_post_group_idx == fc_last_group_idx);

    assign selected_final_output_fire =
        (mode_reg == MODE_FC)
        ?
        fc_final_output_fire
        :
        conv_final_output_fire;


    // ============================================================
    // Active / Mode / Configuration Latch / DONE
    // ============================================================
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            active_reg <=
                1'b0;

            mode_reg <=
                MODE_CONV;

            output_mode_lane_reg <=
                {16{MODE_CONV}};

            output_mode_meta_reg <=
                MODE_CONV;

            pool_enable_reg <=
                1'b0;

            fc1_mode_reg <=
                1'b0;

            conv_cout_cfg_reg <=
                {CFG_C_W{1'b0}};

            conv_out_len_cfg_reg <=
                {CFG_T_W{1'b0}};

            conv_post_out_len_reg <=
                {CFG_T_W{1'b0}};

            fc_out_dim_cfg_reg <=
                {CFG_DIM_W{1'b0}};

            o_done <=
                1'b0;

        end
        else begin

            o_done <=
                1'b0;

            if (start_accept) begin

                active_reg <=
                    1'b1;

                mode_reg <=
                    i_mode;

                output_mode_lane_reg <=
                    {16{i_mode}};

                output_mode_meta_reg <=
                    i_mode;

                pool_enable_reg <=
                    i_pool_enable;

                fc1_mode_reg <=
                    i_fc1_mode;

                conv_cout_cfg_reg <=
                    i_conv_cout;

                conv_out_len_cfg_reg <=
                    i_conv_out_len;

                // C precomputes/registers this before START (see port
                // comment above) - B only latches, no arithmetic here.
                conv_post_out_len_reg <=
                    i_conv_post_out_len;

                fc_out_dim_cfg_reg <=
                    i_fc_out_dim;

            end

            if (selected_final_output_fire) begin

                active_reg <=
                    1'b0;

                o_done <=
                    1'b1;

            end

        end

    end


`ifndef SYNTHESIS

    // ============================================================
    // Simulation-Only Guards
    // ============================================================
    always @(posedge clk) begin

        if (rst_n && start_accept) begin

            if (i_mode == MODE_CONV) begin

                if (
                    (i_conv_cin == 0)
                    ||
                    (i_conv_cout == 0)
                    ||
                    (i_conv_out_len == 0)
                ) begin

                    $display(
                        "[ERROR][UNIFIED] Zero CONV configuration Cin=%0d Cout=%0d OutLen=%0d",
                        i_conv_cin,
                        i_conv_cout,
                        i_conv_out_len
                    );

                end

                if (i_conv_cout[3:0] != 4'b0000) begin

                    $display(
                        "[ERROR][UNIFIED] CONV Cout must be multiple of 16, got %0d",
                        i_conv_cout
                    );

                end

                if ((i_conv_cout >> 4) > MAX_CONV_GROUPS) begin

                    $display(
                        "[ERROR][UNIFIED] CONV groups exceed MAX_CONV_GROUPS: groups=%0d MAX=%0d",
                        (i_conv_cout >> 4),
                        MAX_CONV_GROUPS
                    );

                end

            end
            else begin

                if (
                    (i_fc_in_dim == 0)
                    ||
                    (i_fc_out_dim == 0)
                    ||
                    (i_fc_chunk_count == 0)
                    ||
                    (i_fc_last_valid_taps == 0)
                    ||
                    (i_fc_last_valid_taps > 3)
                ) begin

                    $display(
                        "[ERROR][UNIFIED] Invalid FC configuration IN=%0d OUT=%0d CHUNKS=%0d LAST_TAPS=%0d",
                        i_fc_in_dim,
                        i_fc_out_dim,
                        i_fc_chunk_count,
                        i_fc_last_valid_taps
                    );

                end

            end

        end

    end

`endif

endmodule
