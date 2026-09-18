`timescale 1ns / 1ps

module global_pool_16 #(
    parameter DATA_W     = 16,
    parameter GROUP_W    = 4,
    parameter MAX_GROUPS = 16,
    parameter SUM_W      = 20
)(
    input  wire clk,
    input  wire rst_n,

    // ============================================================
    // Conv9 Input
    //
    // Frozen project format:
    //   [256][17]
    //   signed INT16 Q13
    //   post-ReLU -> nonnegative
    //
    // 16 channels are processed per beat.
    // ============================================================
    input  wire                    i_valid,
    output wire                    o_ready,

    input  wire [16*DATA_W-1:0]    i_data,
    input  wire [15:0]             i_lane_valid_mask,
    input  wire [GROUP_W-1:0]      i_group_idx,

    input  wire                    i_first_time,
    input  wire                    i_last_time,

    // ============================================================
    // Global Pool Output
    //
    // global = gmax + ((sum + 8) // 17)
    //
    // Output:
    //   [256] signed INT16 Q13
    // ============================================================
    output reg                     o_valid,
    input  wire                    i_out_ready,

    output reg [16*DATA_W-1:0]     o_data,
    output reg [15:0]              o_lane_valid_mask,
    output reg [GROUP_W-1:0]       o_group_idx
);


    // ============================================================
    // Widths
    // ============================================================
    localparam DIVIDEND_W = SUM_W + 1;

    // Reciprocal constant:
    //
    // 61681 = 2^16 - 2^12 + 2^8 - 2^4 + 1
    //
    // For the frozen Conv9 domain:
    //
    // 0 <= x <= 32767
    // sum <= 17 * 32767 = 557039
    // n = sum + 8 <= 557047
    //
    // For every integer n in this legal range:
    //
    // floor(n / 17)
    // =
    // floor((n * 61681) / 2^20)
    //
    // Therefore:
    //
    // (n * 61681) >> 20
    //
    // is bit-exact with:
    //
    // n / 17
    //
    // over the complete legal project input domain.
    //
    // This is NOT an approximation.
    // ============================================================
    localparam RECIP_SHIFT = 20;

    // n is DIVIDEND_W bits.
    // n << 16 requires DIVIDEND_W + 16 bits.
    localparam PROD_W =
        DIVIDEND_W + 16;


    // ============================================================
    // Pipeline
    //
    // Stage 1:
    //   final max
    //   n = final sum + 8
    //
    // Stage 2:
    //   A = (n << 16) - (n << 12)
    //   B = (n <<  8) - (n <<  4)
    //
    // Stage 3:
    //   C = A + B
    //
    // Stage 4:
    //   product = C + n
    //   mean    = product >> 20
    //
    // Stage 5:
    //   global = max + mean
    //
    // Mathematically:
    //
    // product
    // =
    // n * (65536 - 4096 + 256 - 16 + 1)
    // =
    // n * 61681
    //
    // Throughput remains one final OC-group beat per cycle.
    // Only latency increases.
    // ============================================================


    // ============================================================
    // Stage 1
    // ============================================================
    reg
        s1_valid;

    reg [16*DATA_W-1:0]
        s1_max;

    reg [16*DIVIDEND_W-1:0]
        s1_dividend;

    reg [15:0]
        s1_lane_valid_mask;

    reg [GROUP_W-1:0]
        s1_group_idx;


    // ============================================================
    // Stage 2
    // ============================================================
    reg
        s2_valid;

    reg [16*DATA_W-1:0]
        s2_max;

    reg [16*PROD_W-1:0]
        s2_part_hi;

    reg [16*PROD_W-1:0]
        s2_part_lo;

    reg [16*DIVIDEND_W-1:0]
        s2_dividend;

    reg [15:0]
        s2_lane_valid_mask;

    reg [GROUP_W-1:0]
        s2_group_idx;


    // ============================================================
    // Stage 3
    // ============================================================
    reg
        s3_valid;

    reg [16*DATA_W-1:0]
        s3_max;

    reg [16*PROD_W-1:0]
        s3_partial_product;

    reg [16*DIVIDEND_W-1:0]
        s3_dividend;

    reg [15:0]
        s3_lane_valid_mask;

    reg [GROUP_W-1:0]
        s3_group_idx;


    // ============================================================
    // Stage 4
    // ============================================================
    reg
        s4_valid;

    reg [16*DATA_W-1:0]
        s4_max;

    reg [16*DATA_W-1:0]
        s4_mean;

    reg [15:0]
        s4_lane_valid_mask;

    reg [GROUP_W-1:0]
        s4_group_idx;


    // ============================================================
    // Elastic Ready Chain
    // ============================================================
    wire output_ready;
    wire s4_ready;
    wire s3_ready;
    wire s2_ready;
    wire s1_ready;


    assign output_ready =
        (~o_valid)
        |
        i_out_ready;


    assign s4_ready =
        (~s4_valid)
        |
        output_ready;


    assign s3_ready =
        (~s3_valid)
        |
        s4_ready;


    assign s2_ready =
        (~s2_valid)
        |
        s3_ready;


    assign s1_ready =
        (~s1_valid)
        |
        s2_ready;


    // Conservative upstream backpressure.
    //
    // All accepted input beats remain ordered.
    assign o_ready =
        s1_ready;


    wire input_fire;

    assign input_fire =
        i_valid
        &&
        o_ready;


    // ============================================================
    // Helper Function:
    // Reciprocal Part HI
    //
    // A = (n << 16) - (n << 12)
    // ============================================================
    function [PROD_W-1:0] reciprocal_part_hi;

        input [DIVIDEND_W-1:0] n;

        reg [PROD_W-1:0]
            n_ext;

        begin

            n_ext =
                {
                    {(PROD_W-DIVIDEND_W){1'b0}},
                    n
                };


            reciprocal_part_hi =
                (n_ext << 16)
                -
                (n_ext << 12);

        end

    endfunction


    // ============================================================
    // Helper Function:
    // Reciprocal Part LO
    //
    // B = (n << 8) - (n << 4)
    // ============================================================
    function [PROD_W-1:0] reciprocal_part_lo;

        input [DIVIDEND_W-1:0] n;

        reg [PROD_W-1:0]
            n_ext;

        begin

            n_ext =
                {
                    {(PROD_W-DIVIDEND_W){1'b0}},
                    n
                };


            reciprocal_part_lo =
                (n_ext << 8)
                -
                (n_ext << 4);

        end

    endfunction


    // ============================================================
    // Helper Function:
    // Complete Exact Divide-by-17
    //
    // partial_product
    // =
    // (n<<16) - (n<<12)
    // +
    // (n<<8)  - (n<<4)
    //
    // product
    // =
    // partial_product + n
    //
    // mean
    // =
    // product >> 20
    // ============================================================
    function [DATA_W-1:0] reciprocal_finish;

        input [PROD_W-1:0]
            partial_product;

        input [DIVIDEND_W-1:0]
            n;

        reg [PROD_W-1:0]
            n_ext;

        reg [PROD_W-1:0]
            product;

        begin

            n_ext =
                {
                    {(PROD_W-DIVIDEND_W){1'b0}},
                    n
                };


            product =
                partial_product
                +
                n_ext;


            reciprocal_finish =
                product
                >>
                RECIP_SHIFT;

        end

    endfunction


    // ============================================================
    // Per-group Running Max / Sum
    //
    // Conv controller ordering:
    //
    // Time outer
    // OC group inner
    //
    // Therefore each OC group needs independent state.
    // ============================================================
    genvar g;

    generate

        for (
            g = 0;
            g < 16;
            g = g + 1
        ) begin : GEN_GLOBAL_STATE

            reg [DATA_W-1:0]
                max_mem [0:MAX_GROUPS-1];

            reg [SUM_W-1:0]
                sum_mem [0:MAX_GROUPS-1];


            wire [DATA_W-1:0]
                input_lane;

            wire [DATA_W-1:0]
                stored_max;

            wire [SUM_W-1:0]
                stored_sum;

            wire [DATA_W-1:0]
                next_max;

            wire [SUM_W-1:0]
                next_sum;


            assign input_lane =
                i_data[
                    g*DATA_W
                    +:
                    DATA_W
                ];


            assign stored_max =
                max_mem[
                    i_group_idx
                ];


            assign stored_sum =
                sum_mem[
                    i_group_idx
                ];


            // ====================================================
            // Running Maximum
            //
            // Conv9 is post-ReLU, so unsigned comparison is valid
            // for its legal nonnegative INT16 Q13 domain.
            // ====================================================
            assign next_max =
                i_first_time
                ?
                input_lane
                :
                (
                    input_lane > stored_max
                    ?
                    input_lane
                    :
                    stored_max
                );


            // ====================================================
            // Running Sum
            //
            // 17 * 32767 = 557039
            // fits within 20 unsigned bits.
            // ====================================================
            assign next_sum =
                i_first_time
                ?
                {
                    {(SUM_W-DATA_W){1'b0}},
                    input_lane
                }
                :
                (
                    stored_sum
                    +
                    {
                        {(SUM_W-DATA_W){1'b0}},
                        input_lane
                    }
                );


            // ====================================================
            // Running State Update
            // ====================================================
            always @(posedge clk) begin

                if (input_fire) begin

                    max_mem[
                        i_group_idx
                    ] <=
                        next_max;


                    sum_mem[
                        i_group_idx
                    ] <=
                        next_sum;

                end

            end


            // ====================================================
            // Stage 1 Data Capture
            //
            // IMPORTANT:
            //
            // next_max / next_sum include the CURRENT input sample.
            // Therefore t=16 is included in the final result.
            // ====================================================
            always @(posedge clk or negedge rst_n) begin

                if (!rst_n) begin

                    s1_max[
                        g*DATA_W
                        +:
                        DATA_W
                    ] <=
                        {DATA_W{1'b0}};


                    s1_dividend[
                        g*DIVIDEND_W
                        +:
                        DIVIDEND_W
                    ] <=
                        {DIVIDEND_W{1'b0}};

                end
                else begin

                    if (
                        input_fire
                        &&
                        i_last_time
                    ) begin

                        s1_max[
                            g*DATA_W
                            +:
                            DATA_W
                        ] <=
                            next_max;


                        // Frozen rule:
                        //
                        // n = sum + 8
                        s1_dividend[
                            g*DIVIDEND_W
                            +:
                            DIVIDEND_W
                        ] <=
                            {
                                1'b0,
                                next_sum
                            }
                            +
                            {{(DIVIDEND_W-4){1'b0}}, 4'd8};

                    end

                end

            end

        end

    endgenerate


    // ============================================================
    // Stage 1 Valid / Metadata
    // ============================================================
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s1_valid <=
                1'b0;

            s1_lane_valid_mask <=
                16'h0000;

            s1_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (s1_ready) begin

                if (
                    input_fire
                    &&
                    i_last_time
                ) begin

                    s1_valid <=
                        1'b1;

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
    // Stage 2
    //
    // Split constant multiplication:
    //
    // A = (n<<16) - (n<<12)
    // B = (n<<8)  - (n<<4)
    //
    // These are independent operations.
    // ============================================================
    integer lane2;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s2_valid <=
                1'b0;

            s2_max <=
                {16*DATA_W{1'b0}};

            s2_part_hi <=
                {16*PROD_W{1'b0}};

            s2_part_lo <=
                {16*PROD_W{1'b0}};

            s2_dividend <=
                {16*DIVIDEND_W{1'b0}};

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

                        s2_part_hi[
                            lane2*PROD_W
                            +:
                            PROD_W
                        ] <=
                            reciprocal_part_hi(
                                s1_dividend[
                                    lane2*DIVIDEND_W
                                    +:
                                    DIVIDEND_W
                                ]
                            );


                        s2_part_lo[
                            lane2*PROD_W
                            +:
                            PROD_W
                        ] <=
                            reciprocal_part_lo(
                                s1_dividend[
                                    lane2*DIVIDEND_W
                                    +:
                                    DIVIDEND_W
                                ]
                            );

                    end


                    s2_max <=
                        s1_max;

                    s2_dividend <=
                        s1_dividend;

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
    // Stage 3
    //
    // partial_product = A + B
    // ============================================================
    integer lane3;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s3_valid <=
                1'b0;

            s3_max <=
                {16*DATA_W{1'b0}};

            s3_partial_product <=
                {16*PROD_W{1'b0}};

            s3_dividend <=
                {16*DIVIDEND_W{1'b0}};

            s3_lane_valid_mask <=
                16'h0000;

            s3_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (s3_ready) begin

                if (s2_valid) begin

                    s3_valid <=
                        1'b1;


                    for (
                        lane3 = 0;
                        lane3 < 16;
                        lane3 = lane3 + 1
                    ) begin

                        s3_partial_product[
                            lane3*PROD_W
                            +:
                            PROD_W
                        ] <=
                            s2_part_hi[
                                lane3*PROD_W
                                +:
                                PROD_W
                            ]
                            +
                            s2_part_lo[
                                lane3*PROD_W
                                +:
                                PROD_W
                            ];

                    end


                    s3_max <=
                        s2_max;

                    s3_dividend <=
                        s2_dividend;

                    s3_lane_valid_mask <=
                        s2_lane_valid_mask;

                    s3_group_idx <=
                        s2_group_idx;

                end
                else begin

                    s3_valid <=
                        1'b0;

                end

            end

        end

    end


    // ============================================================
    // Stage 4
    //
    // product = partial_product + n
    //
    // mean = product >> 20
    //
    // Equivalent to:
    //
    // mean = (sum + 8) // 17
    //
    // over the entire legal Conv9 input range.
    // ============================================================
    integer lane4;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s4_valid <=
                1'b0;

            s4_max <=
                {16*DATA_W{1'b0}};

            s4_mean <=
                {16*DATA_W{1'b0}};

            s4_lane_valid_mask <=
                16'h0000;

            s4_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (s4_ready) begin

                if (s3_valid) begin

                    s4_valid <=
                        1'b1;


                    for (
                        lane4 = 0;
                        lane4 < 16;
                        lane4 = lane4 + 1
                    ) begin

                        s4_mean[
                            lane4*DATA_W
                            +:
                            DATA_W
                        ] <=
                            reciprocal_finish(
                                s3_partial_product[
                                    lane4*PROD_W
                                    +:
                                    PROD_W
                                ],
                                s3_dividend[
                                    lane4*DIVIDEND_W
                                    +:
                                    DIVIDEND_W
                                ]
                            );

                    end


                    s4_max <=
                        s3_max;

                    s4_lane_valid_mask <=
                        s3_lane_valid_mask;

                    s4_group_idx <=
                        s3_group_idx;

                end
                else begin

                    s4_valid <=
                        1'b0;

                end

            end

        end

    end


    // ============================================================
    // Stage 5
    //
    // Frozen operation:
    //
    // global = gmax + gmean
    //
    // Q13 + Q13 -> Q13
    //
    // The A numeric specification does not define a separate
    // Global-Pool saturation operation. Therefore saturation is
    // NOT added here.
    // ============================================================
    integer lane5;

    reg [DATA_W:0]
        global_value;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            o_valid <=
                1'b0;

            o_data <=
                {16*DATA_W{1'b0}};

            o_lane_valid_mask <=
                16'h0000;

            o_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (output_ready) begin

                if (s4_valid) begin

                    o_valid <=
                        1'b1;


                    for (
                        lane5 = 0;
                        lane5 < 16;
                        lane5 = lane5 + 1
                    ) begin

                        global_value =
                            {
                                1'b0,
                                s4_max[
                                    lane5*DATA_W
                                    +:
                                    DATA_W
                                ]
                            }
                            +
                            {
                                1'b0,
                                s4_mean[
                                    lane5*DATA_W
                                    +:
                                    DATA_W
                                ]
                            };


                        o_data[
                            lane5*DATA_W
                            +:
                            DATA_W
                        ] <=
                            global_value[
                                DATA_W-1:0
                            ];


`ifndef SYNTHESIS

                        // ----------------------------------------
                        // The frozen specification defines the
                        // Global output as signed INT16 Q13 but
                        // does not specify a saturation operation.
                        //
                        // Therefore an unexpected overflow is
                        // reported rather than silently saturated.
                        // ----------------------------------------
                        if (
                            global_value
                            >
                            32767
                        ) begin

                            $display(
                                "[ERROR][GLOBAL_POOL] signed INT16 overflow Lane=%0d Value=%0d",
                                lane5,
                                global_value
                            );

                        end

`endif

                    end


                    o_lane_valid_mask <=
                        s4_lane_valid_mask;

                    o_group_idx <=
                        s4_group_idx;

                end
                else begin

                    o_valid <=
                        1'b0;

                end

            end

        end

    end


`ifndef SYNTHESIS

    // ============================================================
    // Legal-domain assertion
    //
    // The exact reciprocal identity used above is guaranteed for
    // the frozen Conv9 project domain.
    //
    // Maximum legal:
    //
    // sum + 8
    // =
    // 17 * 32767 + 8
    // =
    // 557047
    // ============================================================
    integer check_lane;

    always @(posedge clk) begin

        if (
            rst_n
            &&
            s1_valid
        ) begin

            for (
                check_lane = 0;
                check_lane < 16;
                check_lane = check_lane + 1
            ) begin

                if (
                    s1_dividend[
                        check_lane*DIVIDEND_W
                        +:
                        DIVIDEND_W
                    ]
                    >
                    557047
                ) begin

                    $display(
                        "[ERROR][GLOBAL_POOL] Illegal dividend Lane=%0d Value=%0d",
                        check_lane,
                        s1_dividend[
                            check_lane*DIVIDEND_W
                            +:
                            DIVIDEND_W
                        ]
                    );

                end

            end

        end

    end

`endif


endmodule