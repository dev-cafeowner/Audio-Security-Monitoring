`timescale 1ns / 1ps

// Registered observation boundary for CSR error reporting.  Frontend error
// pulses still reach the layer controller directly for same-edge operation
// abort; this block delays only the CSR status write by one clock.
module npu_unified_error_reporter (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_flush,
    input  wire       i_error_pulse,
    input  wire [7:0] i_error_code,
    output wire       o_error_pulse,
    output wire [7:0] o_error_code
);
    reg       error_pulse_reg;
    reg [7:0] error_code_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || i_flush) begin
            error_pulse_reg <= 1'b0;
            error_code_reg  <= 8'h00;
        end else begin
            error_pulse_reg <= i_error_pulse;
            // The pulse is the validity qualifier. Sampling the code every
            // cycle removes a wide CE cone without changing the observable
            // pulse/code pair on an error cycle.
            error_code_reg <= i_error_code;
        end
    end

    assign o_error_pulse = error_pulse_reg;
    assign o_error_code  = error_code_reg;
endmodule
