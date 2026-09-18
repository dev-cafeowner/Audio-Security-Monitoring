`timescale 1ns / 1ps

module accumulator_bias_16 #(
    parameter DATA_W   = 16,
    parameter WEIGHT_W = 16,
    parameter ACC_W    = 48
)(
    input  wire clk,
    input  wire rst_n,

    // One accepted MAC result
    input  wire i_en,

    // First Cin / first FC chunk of current output
    input  wire i_clear,

    // Last Cin / last FC chunk of current output
    input  wire i_last_cin,

    // 16 lanes of 3-product MAC sums
    input  wire [16*(DATA_W+WEIGHT_W+2)-1:0] i_mac_sum,

    // 16 output-channel folded biases
    // Each bias is signed ACC_W (=48) bit
    input  wire [16*ACC_W-1:0] i_bias,

    // Result handshake
    input  wire i_out_ready,

    output wire [16*ACC_W-1:0] o_acc,
    output reg                   o_valid,

    // Can accept another compute step
    output wire                  o_can_compute
);

    localparam MAC_SUM_W =
        DATA_W + WEIGHT_W + 2;

    reg signed [ACC_W-1:0]
        acc_reg [0:15];

    reg signed [ACC_W-1:0]
        result_reg [0:15];

    integer lane;


    // ============================================================
    // Output Packing
    // ============================================================
    genvar g;

    generate
        for (
            g = 0;
            g < 16;
            g = g + 1
        ) begin : GEN_OUTPUT

            assign o_acc[
                g*ACC_W
                +:
                ACC_W
            ] =
                result_reg[g];

        end
    endgenerate


    // ============================================================
    // Result Holding / Backpressure
    // ============================================================
    assign o_can_compute =
        (~o_valid)
        |
        i_out_ready;


    // ============================================================
    // Accumulator
    //
    // A-spec:
    //
    // ACC[oc] =
    //     Bq[oc]
    //     + SUM(Aq * Wq)
    //
    // Bias is added exactly once when i_clear=1.
    //
    // Cin=1 / one FC chunk:
    //
    // i_clear=1 and i_last_cin=1
    //
    // Result =
    //     Bias + current MAC
    // ============================================================
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            o_valid <=
                1'b0;


            for (
                lane = 0;
                lane < 16;
                lane = lane + 1
            ) begin

                acc_reg[lane] <=
                    {ACC_W{1'b0}};

                result_reg[lane] <=
                    {ACC_W{1'b0}};

            end

        end
        else begin

            // ----------------------------------------------------
            // Consume previous held result
            // ----------------------------------------------------
            if (
                o_valid
                &&
                i_out_ready
            ) begin

                o_valid <=
                    1'b0;

            end


            // ----------------------------------------------------
            // Accept compute step
            // ----------------------------------------------------
            if (
                i_en
                &&
                o_can_compute
            ) begin

                for (
                    lane = 0;
                    lane < 16;
                    lane = lane + 1
                ) begin

                    // ============================================
                    // First accumulation step
                    //
                    // Bias + current MAC
                    // ============================================
                    if (i_clear) begin

                        if (i_last_cin) begin

                            result_reg[lane] <=
                                $signed(
                                    i_bias[
                                        lane*ACC_W
                                        +:
                                        ACC_W
                                    ]
                                )
                                +
                                $signed(
                                    i_mac_sum[
                                        lane*MAC_SUM_W
                                        +:
                                        MAC_SUM_W
                                    ]
                                );

                        end
                        else begin

                            acc_reg[lane] <=
                                $signed(
                                    i_bias[
                                        lane*ACC_W
                                        +:
                                        ACC_W
                                    ]
                                )
                                +
                                $signed(
                                    i_mac_sum[
                                        lane*MAC_SUM_W
                                        +:
                                        MAC_SUM_W
                                    ]
                                );

                        end

                    end

                    // ============================================
                    // Following accumulation steps
                    // ============================================
                    else begin

                        if (i_last_cin) begin

                            result_reg[lane] <=
                                $signed(
                                    acc_reg[lane]
                                )
                                +
                                $signed(
                                    i_mac_sum[
                                        lane*MAC_SUM_W
                                        +:
                                        MAC_SUM_W
                                    ]
                                );

                        end
                        else begin

                            acc_reg[lane] <=
                                $signed(
                                    acc_reg[lane]
                                )
                                +
                                $signed(
                                    i_mac_sum[
                                        lane*MAC_SUM_W
                                        +:
                                        MAC_SUM_W
                                    ]
                                );

                        end

                    end

                end


                // ------------------------------------------------
                // Final Cin/chunk creates output
                // ------------------------------------------------
                if (i_last_cin) begin

                    o_valid <=
                        1'b1;

                end

            end

        end

    end

endmodule