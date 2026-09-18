`timescale 1ns / 1ps
`include "npu_defs.vh"

// Immutable runtime bounds for the one unified NPU IP.  This is deliberately
// not a calculator: post-pool length and last-output-group are the FROZEN
// manifest values supplied directly to the frontend/B core.
module npu_unified_descriptor_rom (
    input  wire [3:0] i_layer_id,
    output reg        o_valid,
    output reg [1:0]  o_op_mode,       // 0=Conv, 1=FC, 2=GLOBAL
    output reg        o_pool_enable,
    output reg [`NPU_CFG_C_W-1:0] o_cin,
    output reg [`NPU_CFG_C_W-1:0] o_cout,
    output reg [4:0]  o_num_oc_group,
    output reg [`NPU_CFG_T_W-1:0] o_conv_out_len,
    output reg [`NPU_CFG_T_W-1:0] o_post_out_len,
    output reg [31:0] o_last_output_group,
    // Exact ConvN activation-frame length in 32-bit AXIS beats.  Keeping this
    // manifest-authoritative removes Tconv*Cin/2 from the frontend's runtime
    // configuration path (and the otherwise inferred non-MAC DSP).
    output reg [31:0] o_conv_activation_beats,
    output reg [`NPU_CFG_DIM_W-1:0] o_fc_in_dim,
    output reg [`NPU_CFG_DIM_W-1:0] o_fc_out_dim,
    // FC loop geometry is also immutable descriptor data.  B consumes these
    // fields directly instead of implementing runtime /3 and %3 operators.
    output reg [`NPU_CFG_DIM_W-1:0] o_fc_chunk_count,
    output reg [1:0] o_fc_last_valid_taps
);
    localparam [1:0] OP_CONV   = 2'd0;
    localparam [1:0] OP_FC     = 2'd1;
    localparam [1:0] OP_GLOBAL = 2'd2;

    always @(*) begin
        o_valid             = 1'b1;
        o_op_mode           = OP_CONV;
        o_pool_enable       = 1'b0;
        o_cin               = {`NPU_CFG_C_W{1'b0}};
        o_cout              = {`NPU_CFG_C_W{1'b0}};
        o_num_oc_group      = 5'd0;
        o_conv_out_len      = {`NPU_CFG_T_W{1'b0}};
        o_post_out_len      = {`NPU_CFG_T_W{1'b0}};
        o_last_output_group = 32'd0;
        o_conv_activation_beats = 32'd0;
        o_fc_in_dim         = {`NPU_CFG_DIM_W{1'b0}};
        o_fc_out_dim        = {`NPU_CFG_DIM_W{1'b0}};
        o_fc_chunk_count    = {`NPU_CFG_DIM_W{1'b0}};
        o_fc_last_valid_taps = 2'd0;
        case (i_layer_id)
            // Conv1 retains its dedicated frontend, but shares the B core.
            4'd0: begin o_cin=10'd1;   o_cout=10'd64;  o_num_oc_group=5'd4;  o_conv_out_len=20'd106667; o_post_out_len=20'd106667; o_last_output_group=32'd426667; end
            4'd1: begin o_pool_enable=1'b1; o_cin=10'd64;  o_cout=10'd64;  o_num_oc_group=5'd4;  o_conv_out_len=20'd106667; o_post_out_len=20'd35556;  o_last_output_group=32'd142223; o_conv_activation_beats=32'd3413344; end
            4'd2: begin o_pool_enable=1'b1; o_cin=10'd64;  o_cout=10'd64;  o_num_oc_group=5'd4;  o_conv_out_len=20'd35556;  o_post_out_len=20'd11852;  o_last_output_group=32'd47407;  o_conv_activation_beats=32'd1137792; end
            4'd3: begin o_pool_enable=1'b1; o_cin=10'd64;  o_cout=10'd128; o_num_oc_group=5'd8;  o_conv_out_len=20'd11852;  o_post_out_len=20'd3951;   o_last_output_group=32'd31607;  o_conv_activation_beats=32'd379264;  end
            4'd4: begin o_pool_enable=1'b1; o_cin=10'd128; o_cout=10'd128; o_num_oc_group=5'd8;  o_conv_out_len=20'd3951;   o_post_out_len=20'd1317;   o_last_output_group=32'd10535;  o_conv_activation_beats=32'd252864;  end
            4'd5: begin o_pool_enable=1'b1; o_cin=10'd128; o_cout=10'd128; o_num_oc_group=5'd8;  o_conv_out_len=20'd1317;   o_post_out_len=20'd439;    o_last_output_group=32'd3511;   o_conv_activation_beats=32'd84288;   end
            4'd6: begin o_pool_enable=1'b1; o_cin=10'd128; o_cout=10'd128; o_num_oc_group=5'd8;  o_conv_out_len=20'd439;    o_post_out_len=20'd147;    o_last_output_group=32'd1175;   o_conv_activation_beats=32'd28096;   end
            4'd7: begin o_pool_enable=1'b1; o_cin=10'd128; o_cout=10'd128; o_num_oc_group=5'd8;  o_conv_out_len=20'd147;    o_post_out_len=20'd49;     o_last_output_group=32'd391;    o_conv_activation_beats=32'd9408;    end
            4'd8: begin o_pool_enable=1'b1; o_cin=10'd128; o_cout=10'd256; o_num_oc_group=5'd16; o_conv_out_len=20'd49;     o_post_out_len=20'd17;     o_last_output_group=32'd271;    o_conv_activation_beats=32'd3136;    end
            4'd9: begin o_op_mode=OP_GLOBAL; o_cin=10'd256; o_cout=10'd256; end
            4'd10: begin o_op_mode=OP_FC; o_fc_in_dim=16'd256; o_fc_out_dim=16'd512; o_fc_chunk_count=16'd86;  o_fc_last_valid_taps=2'd1; end
            4'd11: begin o_op_mode=OP_FC; o_fc_in_dim=16'd512; o_fc_out_dim=16'd527; o_fc_chunk_count=16'd171; o_fc_last_valid_taps=2'd2; end
            default: o_valid = 1'b0;
        endcase
    end
endmodule
