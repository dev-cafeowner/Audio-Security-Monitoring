`timescale 1ns / 1ps

module relu48_16 #(
    parameter ACC_W = 48
)(
    input  wire clk,
    input  wire rst_n,

    // ============================================================
    // Input
    // ============================================================
    input  wire                  i_valid,
    output wire                  o_ready,
    input  wire [16*ACC_W-1:0]   i_data,
    input  wire [15:0]           i_lane_valid_mask,

    // ============================================================
    // Output
    // ============================================================
    output reg                   o_valid,
    input  wire                  i_out_ready,
    output reg  [16*ACC_W-1:0]   o_data,
    output reg  [15:0]           o_lane_valid_mask
);

    integer lane;

    reg signed [ACC_W-1:0] lane_value;


    // ============================================================
    // One-entry elastic output register
    // ============================================================
    assign o_ready =
        (~o_valid)
        |
        i_out_ready;


    // ============================================================
    // ReLU
    //
    // y = max(x, 0)
    //
    // IMPORTANT:
    // A-spec requires ReLU before requantization.
    // Input and output therefore remain signed ACC_W (=48) bit.
    // ============================================================
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            o_valid <=
                1'b0;

            o_data <=
                {16*ACC_W{1'b0}};

            o_lane_valid_mask <=
                16'h0000;

        end
        else begin

            // ----------------------------------------------------
            // Consume previous output when no replacement arrives
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
            // Accept new input
            // ----------------------------------------------------
            if (
                i_valid
                &&
                o_ready
            ) begin

                for (
                    lane = 0;
                    lane < 16;
                    lane = lane + 1
                ) begin

                    lane_value =
                        $signed(
                            i_data[
                                lane*ACC_W
                                +:
                                ACC_W
                            ]
                        );


                    if (
                        lane_value[ACC_W-1]
                    ) begin

                        o_data[
                            lane*ACC_W
                            +:
                            ACC_W
                        ] <=
                            {ACC_W{1'b0}};

                    end
                    else begin

                        o_data[
                            lane*ACC_W
                            +:
                            ACC_W
                        ] <=
                            lane_value;

                    end

                end


                o_lane_valid_mask <=
                    i_lane_valid_mask;

                o_valid <=
                    1'b1;

            end

        end

    end

endmodule