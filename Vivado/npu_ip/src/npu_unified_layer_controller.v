`timescale 1ns / 1ps

// Owns a single operation at a time.  The future AXI-Lite CSR supplies the
// request signals; this block makes the START-edge descriptor latch explicit.
module npu_unified_layer_controller (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [3:0] i_layer_id,
    input  wire       i_layer_id_valid,
    input  wire       i_start_pulse,
    input  wire       i_soft_reset_pulse,
    input  wire       i_terminal_done,
    input  wire       i_terminal_error,
    output reg        o_active_valid,
    output reg [3:0]  o_active_layer_id,
    output reg        o_frontend_start_pulse,
    output wire       o_start_accept,
    output wire       o_invalid_start
);
    assign o_start_accept = i_start_pulse && !o_active_valid &&
                            i_layer_id_valid && !i_soft_reset_pulse;
    assign o_invalid_start = i_start_pulse && !o_active_valid &&
                             !i_layer_id_valid && !i_soft_reset_pulse;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_active_valid <= 1'b0;
            o_active_layer_id <= 4'd0;
            o_frontend_start_pulse <= 1'b0;
        end
        else if (i_soft_reset_pulse) begin
            o_active_valid <= 1'b0;
            o_frontend_start_pulse <= 1'b0;
        end
        else begin
            o_frontend_start_pulse <= 1'b0;
            if (o_start_accept) begin
                o_active_valid <= 1'b1;
                o_active_layer_id <= i_layer_id;
                o_frontend_start_pulse <= 1'b1;
            end
            else if (o_active_valid && (i_terminal_done || i_terminal_error)) begin
                o_active_valid <= 1'b0;
            end
        end
    end
endmodule
