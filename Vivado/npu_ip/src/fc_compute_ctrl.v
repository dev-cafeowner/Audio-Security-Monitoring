`timescale 1ns / 1ps

module fc_compute_ctrl #(
    parameter CFG_DIM_W = 10
)(
    input  wire                   clk,
    input  wire                   rst_n,

    input  wire                   i_start,

    // One accepted FC MAC step
    input  wire                   i_step,

    // FC configuration
    input  wire [CFG_DIM_W-1:0]   i_in_dim,
    input  wire [CFG_DIM_W-1:0]   i_out_dim,
    input  wire [CFG_DIM_W-1:0]   i_in_chunk_count,
    input  wire [1:0]             i_last_valid_taps,

    // Compute control
    output wire                   o_compute_en,
    output wire                   o_clear,
    output wire                   o_last_chunk,

    // Current loop indices
    output wire [CFG_DIM_W-1:0]   o_chunk_idx,
    output wire [CFG_DIM_W-1:0]   o_out_group_idx,

    // Number of valid activation values in current 3-value chunk
    // 1 / 2 / 3
    output reg  [1:0]             o_valid_taps,

    // Valid output lanes in current 16-output group
    output reg  [15:0]            o_lane_valid_mask,

    output reg                    o_busy,
    output reg                    o_done
);

    // ============================================================
    // Configuration Registers
    // ============================================================
    reg [CFG_DIM_W-1:0] out_dim_cfg;

    reg [CFG_DIM_W-1:0] in_chunk_count;
    reg [CFG_DIM_W-1:0] out_group_count;
    reg [1:0]           last_valid_taps_cfg;


    // ============================================================
    // Loop Counters
    //
    // Input chunk is inner loop.
    // Output group is outer loop.
    // ============================================================
    reg [CFG_DIM_W-1:0] chunk_idx;
    reg [CFG_DIM_W-1:0] out_group_idx;


    // ============================================================
    // Output remainder.  Input chunk count and final valid-tap count are
    // descriptor-predecoded inputs; no runtime /3 or %3 remains here.
    // ============================================================
    wire [3:0] out_remainder;

    assign out_remainder =
        out_dim_cfg[3:0];


    // ============================================================
    // Outputs
    // ============================================================
    assign o_chunk_idx =
        chunk_idx;

    assign o_out_group_idx =
        out_group_idx;


    assign o_compute_en =
        o_busy &&
        i_step;


    assign o_clear =
        o_busy &&
        i_step &&
        (chunk_idx == 0);


    assign o_last_chunk =
        o_busy &&
        i_step &&
        (
            chunk_idx ==
            (in_chunk_count - 1'b1)
        );


    // ============================================================
    // Current Input-Tap Valid Count
    //
    // Normal chunk : 3
    //
    // Final chunk uses descriptor field i_last_valid_taps (1/2/3).
    // ============================================================
    always @(*) begin

        o_valid_taps =
            2'd3;


        if (
            chunk_idx ==
            (in_chunk_count - 1'b1)
        ) begin

            o_valid_taps = last_valid_taps_cfg;

        end

    end


    // ============================================================
    // Current Output Lane Mask
    //
    // FC2:
    // OutDim = 527
    //
    // 527 = 32*16 + 15
    //
    // Last group:
    // lane 0~14 valid
    // lane 15 invalid
    // mask = 16'h7FFF
    // ============================================================
    always @(*) begin

        o_lane_valid_mask =
            16'hFFFF;


        if (
            out_group_idx ==
            (out_group_count - 1'b1)
        ) begin

            if (
                out_remainder != 0
            ) begin

                o_lane_valid_mask =
                    (16'h0001 << out_remainder)
                    - 1'b1;

            end

        end

    end


    // ============================================================
    // FC Loop Controller
    // ============================================================
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            out_dim_cfg <=
                {CFG_DIM_W{1'b0}};

            in_chunk_count <=
                {CFG_DIM_W{1'b0}};

            last_valid_taps_cfg <=
                2'd0;

            out_group_count <=
                {CFG_DIM_W{1'b0}};

            chunk_idx <=
                {CFG_DIM_W{1'b0}};

            out_group_idx <=
                {CFG_DIM_W{1'b0}};

            o_busy <=
                1'b0;

            o_done <=
                1'b0;

        end
        else begin

            // DONE = one-cycle pulse
            o_done <=
                1'b0;


            // ====================================================
            // IDLE
            // ====================================================
            if (!o_busy) begin

                if (i_start) begin

                    out_dim_cfg <=
                        i_out_dim;


                    // Manifest-authoritative ceil(InDim/3).  Runtime /3 and
                    // %3 were removed from this controller; the descriptor
                    // supplies both loop geometry fields at START.
                    in_chunk_count <=
                        i_in_chunk_count;

                    last_valid_taps_cfg <=
                        i_last_valid_taps;


                    // ceil(OutDim / 16)
                    out_group_count <=
                        (i_out_dim + 15) >> 4;


                    chunk_idx <=
                        {CFG_DIM_W{1'b0}};

                    out_group_idx <=
                        {CFG_DIM_W{1'b0}};

                    o_busy <=
                        1'b1;

                end

            end


            // ====================================================
            // COMPUTE
            // ====================================================
            else begin

                // Only advance on accepted activation + weight.
                if (i_step) begin


                    // =================================================
                    // End of current input-vector chunk loop
                    // =================================================
                    if (
                        chunk_idx ==
                        (in_chunk_count - 1'b1)
                    ) begin

                        chunk_idx <=
                            {CFG_DIM_W{1'b0}};


                        // =============================================
                        // End of current output group
                        // =============================================
                        if (
                            out_group_idx ==
                            (out_group_count - 1'b1)
                        ) begin

                            out_group_idx <=
                                {CFG_DIM_W{1'b0}};

                            o_busy <=
                                1'b0;

                            o_done <=
                                1'b1;

                        end
                        else begin

                            out_group_idx <=
                                out_group_idx + 1'b1;

                        end

                    end


                    // =================================================
                    // Continue same output group
                    // =================================================
                    else begin

                        chunk_idx <=
                            chunk_idx + 1'b1;

                    end

                end

            end

        end

    end

endmodule
