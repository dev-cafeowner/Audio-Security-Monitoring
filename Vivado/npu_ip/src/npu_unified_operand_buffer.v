`timescale 1ns / 1ps
`include "npu_defs.vh"

// One-entry, no-fall-through operand boundary for the legacy Pin1 frontends
// (Conv1 and FC).  Their issue selectors are derived from the B-core dequeue
// state, so allowing the frontend to enqueue a second request before that state
// advances associates the next payload with a stale OC group.  ConvN does not
// use this boundary; it keeps the v5 lane-local Pin4 FIFO and 4-Cin datapath.
module npu_unified_operand_buffer (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         i_flush,

    input  wire                         i_src_valid,
    output wire                         o_src_ready,
    input  wire [3*`NPU_DATA_W-1:0]      i_src_x,
    input  wire [`NPU_WEIGHT_BUS_W-1:0]  i_src_weight,
    input  wire [`NPU_BIAS_BUS_W-1:0]   i_src_bias,

    output wire                         o_b_valid,
    input  wire                         i_b_act_ready,
    input  wire                         i_b_weight_ready,
    output wire [3*`NPU_DATA_W-1:0]      o_b_x,
    output wire [`NPU_WEIGHT_BUS_W-1:0]  o_b_weight,
    output wire [`NPU_BIAS_BUS_W-1:0]   o_b_bias
);
    reg                            valid_reg;
    reg [3*`NPU_DATA_W-1:0]        x_reg;
    reg [`NPU_WEIGHT_BUS_W-1:0]    weight_reg;
    reg [`NPU_BIAS_BUS_W-1:0]      bias_reg;

    wire src_enqueue_fire = i_src_valid && o_src_ready;
    wire b_dequeue_fire = o_b_valid && i_b_act_ready && i_b_weight_ready;

    // Deliberately do not accept a replacement on the dequeue cycle.  The
    // frontend observes ready again on the following cycle, after the B-core
    // group/time state has advanced.
    assign o_src_ready = !valid_reg;
    assign o_b_valid   = valid_reg;
    assign o_b_x       = x_reg;
    assign o_b_weight  = weight_reg;
    assign o_b_bias    = bias_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || i_flush) begin
            valid_reg <= 1'b0;
        end else begin
            if (src_enqueue_fire) begin
                x_reg      <= i_src_x;
                weight_reg <= i_src_weight;
                bias_reg   <= i_src_bias;
                valid_reg  <= 1'b1;
            end else if (b_dequeue_fire) begin
                valid_reg <= 1'b0;
            end
        end
    end
endmodule
