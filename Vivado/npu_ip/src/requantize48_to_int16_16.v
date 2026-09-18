`timescale 1ns / 1ps

module requantize48_to_int16_16 #(
    parameter ACC_W   = 48,
    parameter OUT_W   = 16,
    parameter SHIFT_W = 8,
    parameter GROUP_W = 4
)(
    input  wire clk,
    input  wire rst_n,

    // ============================================================
    // Input
    // ============================================================
    input  wire                      i_valid,
    output wire                      o_ready,

    input  wire [16*ACC_W-1:0]       i_data,
    input  wire [16*SHIFT_W-1:0]     i_rshift,

    input  wire [15:0]               i_lane_valid_mask,
    input  wire [GROUP_W-1:0]        i_group_idx,

    // ============================================================
    // Output
    // ============================================================
    output reg                       o_valid,
    input  wire                      i_out_ready,

    output reg [16*OUT_W-1:0]        o_data,
    output reg [15:0]                o_lane_valid_mask,
    output reg [GROUP_W-1:0]         o_group_idx
);


    // ============================================================
    // Constants
    // ============================================================
    localparam MAG_W =
        ACC_W + 1;

    localparam [OUT_W-1:0] POS_MAX =
        {1'b0, {(OUT_W-1){1'b1}}};

    localparam [OUT_W-1:0] NEG_MAX_MAG =
        {1'b1, {(OUT_W-1){1'b0}}};


    // ============================================================
    // Absolute Magnitude
    //
    // ACC_W+1 bits are used so signed48 minimum:
    //
    // -2^47
    //
    // can be represented as magnitude +2^47.
    // ============================================================
    function [MAG_W-1:0] abs_magnitude;

        input signed [ACC_W-1:0] x;

        begin

            if (x[ACC_W-1]) begin

                abs_magnitude =
                    {1'b0, (~x)}
                    +
                    {{ACC_W{1'b0}}, 1'b1};

            end
            else begin

                abs_magnitude =
                    {1'b0, x};

            end

        end

    endfunction


    // ============================================================
    // Shifted Integer Part
    //
    // q = magnitude >> s
    // ============================================================
    function [MAG_W-1:0] shifted_base;

        input [MAG_W-1:0] magnitude;
        input [SHIFT_W-1:0] shift_value;

        begin

            shifted_base =
                magnitude
                >>
                shift_value;

        end

    endfunction


    // ============================================================
    // Runtime Round Bit
    //
    // For positive magnitude:
    //
    // round(magnitude / 2^s)
    //
    // with nearest / ties-away:
    //
    // q         = magnitude >> s
    // round_bit = magnitude[s-1]
    // rounded   = q + round_bit
    //
    // This is bit-exact with:
    //
    // (magnitude + 2^(s-1)) >> s
    //
    // for s > 0.
    // ============================================================
    function rounding_bit;

        input [MAG_W-1:0] magnitude;
        input [SHIFT_W-1:0] shift_value;

        begin

            if (shift_value == 0) begin

                rounding_bit =
                    1'b0;

            end
            else if (
                shift_value
                <=
                MAG_W
            ) begin

                rounding_bit =
                    magnitude[
                        shift_value - 1'b1
                    ];

            end
            else begin

                // magnitude is only MAG_W bits.
                // For a shift larger than MAG_W,
                // the half threshold cannot be reached.
                rounding_bit =
                    1'b0;

            end

        end

    endfunction


    // ============================================================
    // Final Round + Saturation + Sign Restore
    //
    // IMPORTANT:
    //
    // Instead of performing another full-width 49-bit addition,
    // saturation is decided from:
    //
    //     q
    //     round_bit
    //
    // Then only the low OUT_W bits require an increment.
    //
    // Numeric result is unchanged.
    // ============================================================
    function [OUT_W-1:0] finalize_one;

        input [MAG_W-1:0] q_value;
        input             round_bit;
        input             sign_bit;

        reg [MAG_W-1:0]
            pos_limit_ext;

        reg [MAG_W-1:0]
            neg_limit_ext;

        reg [OUT_W:0]
            small_mag;

        begin

            pos_limit_ext =
                {
                    {(MAG_W-OUT_W){1'b0}},
                    POS_MAX
                };


            neg_limit_ext =
                {
                    {(MAG_W-OUT_W){1'b0}},
                    NEG_MAX_MAG
                };


            // ====================================================
            // Positive input
            // ====================================================
            if (!sign_bit) begin

                // Rounded result would exceed +32767
                if (
                    (q_value > pos_limit_ext)
                    ||
                    (
                        (q_value == pos_limit_ext)
                        &&
                        round_bit
                    )
                ) begin

                    finalize_one =
                        POS_MAX;

                end
                else begin

                    small_mag =
                        {1'b0, q_value[OUT_W-1:0]}
                        +
                        round_bit;


                    finalize_one =
                        small_mag[
                            OUT_W-1:0
                        ];

                end

            end

            // ====================================================
            // Negative input
            // ====================================================
            else begin

                // Legal magnitude limit for INT16 negative:
                //
                // 32768 -> -32768
                //
                // q=32768 + round_bit=1 means 32769
                // and must saturate.
                if (
                    (q_value > neg_limit_ext)
                    ||
                    (
                        (q_value == neg_limit_ext)
                        &&
                        round_bit
                    )
                ) begin

                    finalize_one =
                        NEG_MAX_MAG;

                end
                else begin

                    small_mag =
                        {1'b0, q_value[OUT_W-1:0]}
                        +
                        round_bit;


                    // Restore negative sign using two's complement.
                    finalize_one =
                        (~small_mag[OUT_W-1:0])
                        +
                        {{(OUT_W-1){1'b0}}, 1'b1};

                end

            end

        end

    endfunction


    // ============================================================
    // Pipeline
    //
    // Stage 1
    //     Sign + absolute magnitude
    //
    // Stage 2
    //     Variable right shift + round bit extraction
    //
    // Stage 3
    //     Round increment + saturation + sign restoration
    //
    // Every stage is elastic:
    // throughput = 1 beat / cycle
    // ============================================================


    // ============================================================
    // STAGE 1
    // ============================================================
    reg
        s1_valid;

    reg [16*MAG_W-1:0]
        s1_magnitude;

    reg [15:0]
        s1_sign;

    reg [16*SHIFT_W-1:0]
        s1_rshift;

    reg [15:0]
        s1_lane_valid_mask;

    reg [GROUP_W-1:0]
        s1_group_idx;


    // ============================================================
    // STAGE 2
    // ============================================================
    reg
        s2_valid;

    reg [16*MAG_W-1:0]
        s2_base;

    reg [15:0]
        s2_round_bit;

    reg [15:0]
        s2_sign;

    reg [15:0]
        s2_lane_valid_mask;

    reg [GROUP_W-1:0]
        s2_group_idx;


    // ============================================================
    // Elastic Ready Chain
    // ============================================================
    wire s3_ready;
    wire s2_ready;
    wire s1_ready;


    assign s3_ready =
        (~o_valid)
        |
        i_out_ready;


    assign s2_ready =
        (~s2_valid)
        |
        s3_ready;


    assign s1_ready =
        (~s1_valid)
        |
        s2_ready;


    assign o_ready =
        s1_ready;


    // ============================================================
    // STAGE 1
    // Signed ACC -> Sign + Magnitude
    // ============================================================
    integer lane1;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s1_valid <=
                1'b0;

            s1_magnitude <=
                {16*MAG_W{1'b0}};

            s1_sign <=
                16'h0000;

            s1_rshift <=
                {16*SHIFT_W{1'b0}};

            s1_lane_valid_mask <=
                16'h0000;

            s1_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (s1_ready) begin

                if (i_valid) begin

                    s1_valid <=
                        1'b1;


                    for (
                        lane1 = 0;
                        lane1 < 16;
                        lane1 = lane1 + 1
                    ) begin

                        s1_magnitude[
                            lane1*MAG_W
                            +:
                            MAG_W
                        ] <=
                            abs_magnitude(
                                $signed(
                                    i_data[
                                        lane1*ACC_W
                                        +:
                                        ACC_W
                                    ]
                                )
                            );


                        s1_sign[
                            lane1
                        ] <=
                            i_data[
                                lane1*ACC_W
                                +
                                ACC_W
                                -
                                1
                            ];

                    end


                    s1_rshift <=
                        i_rshift;

                    s1_lane_valid_mask <=
                        i_lane_valid_mask;

                    s1_group_idx <=
                        i_group_idx;

                end
                else begin

                    s1_valid <=
                        1'b0;

                end

            end

        end

    end


    // ============================================================
    // STAGE 2
    // Magnitude -> Shifted base + rounding bit
    // ============================================================
    integer lane2;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s2_valid <=
                1'b0;

            s2_base <=
                {16*MAG_W{1'b0}};

            s2_round_bit <=
                16'h0000;

            s2_sign <=
                16'h0000;

            s2_lane_valid_mask <=
                16'h0000;

            s2_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (s2_ready) begin

                if (s1_valid) begin

                    s2_valid <=
                        1'b1;


                    for (
                        lane2 = 0;
                        lane2 < 16;
                        lane2 = lane2 + 1
                    ) begin

                        s2_base[
                            lane2*MAG_W
                            +:
                            MAG_W
                        ] <=
                            shifted_base(
                                s1_magnitude[
                                    lane2*MAG_W
                                    +:
                                    MAG_W
                                ],
                                s1_rshift[
                                    lane2*SHIFT_W
                                    +:
                                    SHIFT_W
                                ]
                            );


                        s2_round_bit[
                            lane2
                        ] <=
                            rounding_bit(
                                s1_magnitude[
                                    lane2*MAG_W
                                    +:
                                    MAG_W
                                ],
                                s1_rshift[
                                    lane2*SHIFT_W
                                    +:
                                    SHIFT_W
                                ]
                            );

                    end


                    s2_sign <=
                        s1_sign;

                    s2_lane_valid_mask <=
                        s1_lane_valid_mask;

                    s2_group_idx <=
                        s1_group_idx;

                end
                else begin

                    s2_valid <=
                        1'b0;

                end

            end

        end

    end


    // ============================================================
    // STAGE 3
    // Round + Saturate + Sign Restore
    // ============================================================
    integer lane3;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            o_valid <=
                1'b0;

            o_data <=
                {16*OUT_W{1'b0}};

            o_lane_valid_mask <=
                16'h0000;

            o_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (s3_ready) begin

                if (s2_valid) begin

                    o_valid <=
                        1'b1;


                    for (
                        lane3 = 0;
                        lane3 < 16;
                        lane3 = lane3 + 1
                    ) begin

                        o_data[
                            lane3*OUT_W
                            +:
                            OUT_W
                        ] <=
                            finalize_one(
                                s2_base[
                                    lane3*MAG_W
                                    +:
                                    MAG_W
                                ],
                                s2_round_bit[
                                    lane3
                                ],
                                s2_sign[
                                    lane3
                                ]
                            );

                    end


                    o_lane_valid_mask <=
                        s2_lane_valid_mask;

                    o_group_idx <=
                        s2_group_idx;

                end
                else begin

                    o_valid <=
                        1'b0;

                end

            end

        end

    end

endmodule