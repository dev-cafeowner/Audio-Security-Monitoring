`timescale 1ns / 1ps

// Unified v5 Pin4 compute core.
//
// CONV accepts four adjacent Cin channels per step. Conv1 and FC reuse the
// same physical 192-DSP array with only Pin0 enabled.
//
// v7.0 keeps the conservative group barrier for the short legacy Conv1/FC
// transactions, but pipelines Conv2..Conv9 output groups. Their minimum
// Cin=64 gives at least sixteen Pin4 steps between consecutive results, which
// is longer than the four-cycle MAC pipeline. The next group's clear-tagged
// first result can therefore follow the previous group's last result without
// an accumulator hazard. A held output still blocks new input immediately;
// the already in-flight MACs cannot reach another group result before that
// backpressure has propagated because of the same minimum group length.
module shared_mac_acc_core #(
    parameter DATA_W    = 16,
    parameter WEIGHT_W  = 16,
    parameter ACC_W     = 48,
    parameter CFG_C_W   = 10,
    parameter CFG_T_W   = 20,
    parameter CFG_DIM_W = 10
)(
    input  wire clk,
    input  wire rst_n,
    input  wire i_mode,
    input  wire i_start,
    input  wire [CFG_C_W-1:0] i_conv_cin,
    input  wire [CFG_C_W-1:0] i_conv_cout,
    input  wire [CFG_T_W-1:0] i_conv_out_len,
    input  wire [CFG_DIM_W-1:0] i_fc_in_dim,
    input  wire [CFG_DIM_W-1:0] i_fc_out_dim,
    input  wire [CFG_DIM_W-1:0] i_fc_chunk_count,
    input  wire [1:0] i_fc_last_valid_taps,

    input  wire i_act_valid,
    output wire o_act_ready,
    input  wire [4*3*DATA_W-1:0] i_x,
    input  wire i_weight_valid,
    output wire o_weight_ready,
    input  wire [16*4*3*WEIGHT_W-1:0] i_weight,
    input  wire [16*ACC_W-1:0] i_bias,

    input  wire i_result_ready,
    output wire o_result_valid,
    output wire [16*ACC_W-1:0] o_result,
    output wire [15:0] o_lane_valid_mask,
    output wire o_busy,
    output reg  o_done,

    output wire [CFG_C_W-1:0] o_conv_cin_idx,
    output wire [CFG_C_W-1:0] o_conv_oc_group_idx,
    output wire [CFG_T_W-1:0] o_conv_time_idx,
    output wire [CFG_DIM_W-1:0] o_fc_chunk_idx,
    output wire [CFG_DIM_W-1:0] o_fc_out_group_idx
);
    localparam MODE_CONV = 1'b0;
    localparam MODE_FC   = 1'b1;
    localparam PIN       = 4;
    localparam TAPS      = 3;
    localparam MAC_SUM_W = DATA_W + WEIGHT_W + 4;

    reg mode_reg;
    reg [CFG_C_W-1:0] conv_cin_cfg;
    reg [CFG_C_W-1:0] conv_cout_cfg;
    reg [CFG_T_W-1:0] conv_out_len_cfg;
    reg [CFG_DIM_W-1:0] fc_out_dim_cfg;

    wire conv_busy;
    wire fc_busy;
    wire conv_clear;
    wire conv_last_cin;
    wire fc_clear;
    wire fc_last_chunk;
    wire [1:0] fc_valid_taps;
    wire [15:0] fc_lane_valid_mask;
    wire active_ctrl_busy = (mode_reg == MODE_FC) ? fc_busy : conv_busy;

    // LeeNet11 Conv2..Conv9 all have Cin>=64. Keep a conservative threshold
    // of 32 channels (eight Pin4 steps) so the fixed four-cycle pipeline has
    // explicit response/backpressure separation. Conv1 and FC retain the
    // frozen v6 one-group-at-a-time contract.
    wire streaming_conv = (mode_reg == MODE_CONV) &&
                          (conv_cin_cfg >= {{(CFG_C_W-6){1'b0}},6'd32});

    reg group_block;
    reg pending_layer_last;
    reg [15:0] pending_lane_mask;
    reg result_valid_reg;
    reg [15:0] result_lane_mask_reg;
    reg signed [ACC_W-1:0] accumulator [0:15];
    reg signed [ACC_W-1:0] result_reg [0:15];

    wire result_fire = result_valid_reg && i_result_ready;
    assign o_busy = active_ctrl_busy || group_block || result_valid_reg;

    wire conv_start = i_start && !o_busy && (i_mode == MODE_CONV);
    wire fc_start   = i_start && !o_busy && (i_mode == MODE_FC);
    wire streaming_result_stall = streaming_conv && result_valid_reg &&
                                  !i_result_ready;
    wire input_block = group_block || streaming_result_stall;
    wire input_fire = active_ctrl_busy && !input_block &&
                      i_act_valid && i_weight_valid;

    assign o_act_ready = active_ctrl_busy && !input_block && i_weight_valid;
    assign o_weight_ready = active_ctrl_busy && !input_block && i_act_valid;

    wire selected_clear = (mode_reg == MODE_FC) ? fc_clear : conv_clear;
    wire selected_last  = (mode_reg == MODE_FC) ? fc_last_chunk : conv_last_cin;
    wire [15:0] selected_lane_mask =
        (mode_reg == MODE_FC) ? fc_lane_valid_mask : 16'hFFFF;

    wire conv_layer_last = conv_last_cin &&
        (o_conv_oc_group_idx == ((conv_cout_cfg >> 4) - 1'b1)) &&
        (o_conv_time_idx == (conv_out_len_cfg - 1'b1));
    wire fc_layer_last = fc_last_chunk &&
        (o_fc_out_group_idx == (((fc_out_dim_cfg + 15) >> 4) - 1'b1));
    wire selected_layer_last =
        (mode_reg == MODE_FC) ? fc_layer_last : conv_layer_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_reg <= MODE_CONV;
            conv_cin_cfg <= {CFG_C_W{1'b0}};
            conv_cout_cfg <= {CFG_C_W{1'b0}};
            conv_out_len_cfg <= {CFG_T_W{1'b0}};
            fc_out_dim_cfg <= {CFG_DIM_W{1'b0}};
        end else if (i_start && !o_busy) begin
            mode_reg <= i_mode;
            if (i_mode == MODE_CONV) begin
                conv_cin_cfg <= i_conv_cin;
                conv_cout_cfg <= i_conv_cout;
                conv_out_len_cfg <= i_conv_out_len;
            end else begin
                fc_out_dim_cfg <= i_fc_out_dim;
            end
        end
    end

    conv_compute_ctrl #(.CFG_C_W(CFG_C_W),.CFG_T_W(CFG_T_W))
    u_conv_compute_ctrl (
        .clk(clk),.rst_n(rst_n),.i_start(conv_start),.i_step(input_fire),
        .i_cin(i_conv_cin),.i_cout(i_conv_cout),.i_out_len(i_conv_out_len),
        .o_compute_en(),.o_clear(conv_clear),.o_last_cin(conv_last_cin),
        .o_cin_idx(o_conv_cin_idx),.o_oc_group_idx(o_conv_oc_group_idx),
        .o_time_idx(o_conv_time_idx),.o_busy(conv_busy),.o_done()
    );

    fc_compute_ctrl #(.CFG_DIM_W(CFG_DIM_W)) u_fc_compute_ctrl (
        .clk(clk),.rst_n(rst_n),.i_start(fc_start),.i_step(input_fire),
        .i_in_dim(i_fc_in_dim),.i_out_dim(i_fc_out_dim),
        .i_in_chunk_count(i_fc_chunk_count),
        .i_last_valid_taps(i_fc_last_valid_taps),.o_compute_en(),
        .o_clear(fc_clear),.o_last_chunk(fc_last_chunk),
        .o_chunk_idx(o_fc_chunk_idx),.o_out_group_idx(o_fc_out_group_idx),
        .o_valid_taps(fc_valid_taps),.o_lane_valid_mask(fc_lane_valid_mask),
        .o_busy(fc_busy),.o_done()
    );

    // Conv masks channels beyond Cin (Conv1); FC uses only Pin0 and retains
    // the existing tail-tap masking contract.
    wire [PIN*TAPS*DATA_W-1:0] x_masked;
    genvar xp, xt;
    generate
        for (xp=0; xp<PIN; xp=xp+1) begin : GEN_X_PIN
            for (xt=0; xt<TAPS; xt=xt+1) begin : GEN_X_TAP
                wire conv_channel_valid =
                    ((o_conv_cin_idx + xp) < conv_cin_cfg);
                wire fc_tap_valid = (xp == 0) &&
                    ((xt == 0) || ((xt == 1) && (fc_valid_taps >= 2)) ||
                                   ((xt == 2) && (fc_valid_taps >= 3)));
                assign x_masked[((xp*TAPS+xt)*DATA_W) +: DATA_W] =
                    (mode_reg == MODE_FC) ?
                        (fc_tap_valid ? i_x[((xp*TAPS+xt)*DATA_W) +: DATA_W] : {DATA_W{1'b0}}) :
                        (conv_channel_valid ? i_x[((xp*TAPS+xt)*DATA_W) +: DATA_W] : {DATA_W{1'b0}});
            end
        end
    endgenerate

    wire mac_valid;
    wire [16*MAC_SUM_W-1:0] mac_sum;
    mac_array_16x3x4 #(.DATA_W(DATA_W),.WEIGHT_W(WEIGHT_W))
    u_shared_mac_array_16x3x4 (
        .clk(clk),.rst_n(rst_n),.i_valid(input_fire),.i_x(x_masked),
        .i_weight(i_weight),.o_valid(mac_valid),.o_mac_sum(mac_sum)
    );

    // Four-cycle metadata pipeline aligned to mac_array_16x3x4.o_valid.
    reg clear_s0,clear_s1,clear_s2,clear_s3;
    reg last_s0,last_s1,last_s2,last_s3;
    reg signed [ACC_W-1:0] bias_s0 [0:15];
    reg signed [ACC_W-1:0] bias_s1 [0:15];
    reg signed [ACC_W-1:0] bias_s2 [0:15];
    reg signed [ACC_W-1:0] bias_s3 [0:15];
    integer lane_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clear_s0<=0; clear_s1<=0; clear_s2<=0; clear_s3<=0;
            last_s0<=0;  last_s1<=0;  last_s2<=0;  last_s3<=0;
            group_block<=0; pending_layer_last<=0; pending_lane_mask<=0;
            result_valid_reg<=0; result_lane_mask_reg<=0; o_done<=0;
            for (lane_i=0; lane_i<16; lane_i=lane_i+1) begin
                bias_s0[lane_i]<=0; bias_s1[lane_i]<=0;
                bias_s2[lane_i]<=0; bias_s3[lane_i]<=0;
                accumulator[lane_i]<=0; result_reg[lane_i]<=0;
            end
        end else begin
            clear_s0 <= input_fire && selected_clear;
            clear_s1 <= clear_s0; clear_s2 <= clear_s1; clear_s3 <= clear_s2;
            last_s0 <= input_fire && selected_last;
            last_s1 <= last_s0; last_s2 <= last_s1; last_s3 <= last_s2;
            o_done <= 1'b0;

            if (input_fire)
                for (lane_i=0; lane_i<16; lane_i=lane_i+1)
                    bias_s0[lane_i] <= i_bias[lane_i*ACC_W +: ACC_W];
            for (lane_i=0; lane_i<16; lane_i=lane_i+1) begin
                bias_s1[lane_i] <= bias_s0[lane_i];
                bias_s2[lane_i] <= bias_s1[lane_i];
                bias_s3[lane_i] <= bias_s2[lane_i];
            end

            if (input_fire && selected_last) begin
                // Intermediate ConvN groups stream directly into the MAC
                // pipeline. The final group still raises the barrier so
                // o_busy remains asserted through pipeline drain and result
                // acceptance. Legacy Conv1/FC always use the barrier.
                if (!streaming_conv || selected_layer_last)
                    group_block <= 1'b1;
                pending_layer_last <= selected_layer_last;
                pending_lane_mask <= selected_lane_mask;
            end
            if (result_fire) begin
                group_block <= 1'b0;
                result_valid_reg <= 1'b0;
                if (pending_layer_last)
                    o_done <= 1'b1;
            end

            if (mac_valid) begin
                for (lane_i=0; lane_i<16; lane_i=lane_i+1) begin
                    if (clear_s3) begin
                        accumulator[lane_i] <=
                            $signed(bias_s3[lane_i]) +
                            $signed({{(ACC_W-MAC_SUM_W){mac_sum[lane_i*MAC_SUM_W+MAC_SUM_W-1]}},
                                     mac_sum[lane_i*MAC_SUM_W +: MAC_SUM_W]});
                        if (last_s3)
                            result_reg[lane_i] <=
                                $signed(bias_s3[lane_i]) +
                                $signed({{(ACC_W-MAC_SUM_W){mac_sum[lane_i*MAC_SUM_W+MAC_SUM_W-1]}},
                                         mac_sum[lane_i*MAC_SUM_W +: MAC_SUM_W]});
                    end else begin
                        accumulator[lane_i] <=
                            $signed(accumulator[lane_i]) +
                            $signed({{(ACC_W-MAC_SUM_W){mac_sum[lane_i*MAC_SUM_W+MAC_SUM_W-1]}},
                                     mac_sum[lane_i*MAC_SUM_W +: MAC_SUM_W]});
                        if (last_s3)
                            result_reg[lane_i] <=
                                $signed(accumulator[lane_i]) +
                                $signed({{(ACC_W-MAC_SUM_W){mac_sum[lane_i*MAC_SUM_W+MAC_SUM_W-1]}},
                                         mac_sum[lane_i*MAC_SUM_W +: MAC_SUM_W]});
                    end
                end
                if (last_s3) begin
                    result_valid_reg <= 1'b1;
                    result_lane_mask_reg <= pending_lane_mask;
                end
            end
        end
    end

    genvar ro;
    generate
        for (ro=0; ro<16; ro=ro+1)
            assign o_result[ro*ACC_W +: ACC_W] = result_reg[ro];
    endgenerate
    assign o_result_valid = result_valid_reg;
    assign o_lane_valid_mask = result_lane_mask_reg;
endmodule
