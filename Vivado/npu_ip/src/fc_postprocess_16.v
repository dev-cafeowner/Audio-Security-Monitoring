`timescale 1ns / 1ps

module fc_postprocess_16 #(
    parameter ACC_W   = 48,
    parameter OUT_W   = 16,
    parameter SHIFT_W = 8,

    // FC2:
    // ceil(527 / 16) = 33 groups
    //
    // Therefore group index 0~32 requires 6 bits.
    parameter GROUP_W = 6
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // ============================================================
    // Input from Shared MAC/ACC Core
    //
    // Each beat contains final ACC values for 16 output neurons.
    //
    // FC1:
    //   [256] Q13
    //      ↓
    //   FC 256 -> 512
    //
    // FC2:
    //   [512] Q11
    //      ↓
    //   FC 512 -> 527
    //
    // i_acc already includes:
    //
    //   Folded Bias
    //   +
    //   all MAC products
    //
    // signed 48-bit per lane.
    // ============================================================
    input  wire                     i_valid,
    output wire                     o_ready,

    input  wire [16*ACC_W-1:0]      i_acc,

    // Per-output-channel runtime Rshift.
    //
    // FC1 actual range : 15~20
    // FC2 actual range : 15~17
    input  wire [16*SHIFT_W-1:0]    i_rshift,

    input  wire [15:0]              i_lane_valid_mask,
    input  wire [GROUP_W-1:0]       i_group_idx,

    // ============================================================
    // Internal B control
    //
    // 1 = FC1:
    //       ReLU -> Requant -> Saturation
    //
    // 0 = FC2:
    //       Requant -> Saturation
    //
    // This signal is an implementation-detail interface.
    // The arithmetic behavior is frozen by A's specification.
    // ============================================================
    input  wire                     i_fc1_mode,

    // ============================================================
    // INT16 FC Output
    //
    // FC1 -> Q11
    // FC2 -> Q10
    //
    // Q-format itself is determined by the supplied per-OC Rshift.
    // ============================================================
    output wire                     o_valid,
    input  wire                     i_out_ready,

    output wire [16*OUT_W-1:0]      o_data,
    output wire [15:0]              o_lane_valid_mask,
    output wire [GROUP_W-1:0]       o_group_idx
);


    // ============================================================
    // Pre-Requant Elastic Register
    //
    // This stage performs:
    //
    // FC1 : ReLU on signed48 ACC
    // FC2 : direct ACC bypass
    //
    // It also registers Rshift / mask / group together so the
    // metadata stays aligned with the corresponding ACC data.
    // ============================================================
    reg                         preq_valid;

    reg [16*ACC_W-1:0]          preq_data;
    reg [16*SHIFT_W-1:0]        preq_rshift;

    reg [15:0]                  preq_lane_valid_mask;
    reg [GROUP_W-1:0]           preq_group_idx;


    // ============================================================
    // Requant Input Ready
    // ============================================================
    wire rq_ready;


    // ============================================================
    // Elastic Ready
    //
    // The pre-Requant stage can accept a new beat when:
    //
    // 1. it is currently empty
    // OR
    // 2. its current beat is accepted by the requant pipeline
    //
    // This preserves 1 beat/cycle throughput.
    // ============================================================
    wire preq_ready;

    assign preq_ready =
        (~preq_valid)
        |
        rq_ready;


    assign o_ready =
        preq_ready;


    wire input_fire;

    assign input_fire =
        i_valid
        &&
        o_ready;


    // ============================================================
    // FC1 ReLU / FC2 Bypass
    //
    // FC1:
    //
    //   x < 0 ? 0 : x
    //
    // FC2:
    //
    //   x
    //
    // Because ACC is signed48, bit 47 is the sign bit.
    // ============================================================
    integer lane;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            preq_valid <=
                1'b0;

            preq_data <=
                {16*ACC_W{1'b0}};

            preq_rshift <=
                {16*SHIFT_W{1'b0}};

            preq_lane_valid_mask <=
                16'h0000;

            preq_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (preq_ready) begin

                if (input_fire) begin

                    preq_valid <=
                        1'b1;


                    // ------------------------------------------------
                    // Numeric payload
                    // ------------------------------------------------
                    for (
                        lane = 0;
                        lane < 16;
                        lane = lane + 1
                    ) begin

                        // ============================================
                        // FC1 only:
                        // signed48 ReLU BEFORE requantization.
                        // ============================================
                        if (
                            i_fc1_mode
                            &&
                            i_acc[
                                lane*ACC_W
                                +
                                ACC_W
                                -
                                1
                            ]
                        ) begin

                            preq_data[
                                lane*ACC_W
                                +:
                                ACC_W
                            ] <=
                                {ACC_W{1'b0}};

                        end
                        else begin

                            // ========================================
                            // FC1 positive path
                            // OR
                            // FC2 direct path
                            // ========================================
                            preq_data[
                                lane*ACC_W
                                +:
                                ACC_W
                            ] <=
                                i_acc[
                                    lane*ACC_W
                                    +:
                                    ACC_W
                                ];

                        end

                    end


                    // ------------------------------------------------
                    // Metadata
                    // ------------------------------------------------
                    preq_rshift <=
                        i_rshift;

                    preq_lane_valid_mask <=
                        i_lane_valid_mask;

                    preq_group_idx <=
                        i_group_idx;

                end
                else begin

                    preq_valid <=
                        1'b0;

                end

            end

        end

    end


    // ============================================================
    // Existing Bit-Exact Requantizer
    //
    // Frozen runtime rule:
    //
    // x >= 0:
    //   y = (x + 2^(s-1)) >> s
    //
    // x < 0:
    //   y = -(((-x) + 2^(s-1)) >> s)
    //
    // followed by:
    //
    // signed INT16 saturation.
    //
    // No FC-specific rounding implementation is added here.
    // The already verified common requant module is reused.
    // ============================================================
    requantize48_to_int16_16 #(
        .ACC_W   (ACC_W),
        .OUT_W   (OUT_W),
        .SHIFT_W (SHIFT_W),
        .GROUP_W (GROUP_W)
    ) u_requantize48_to_int16_16 (
        .clk               (clk),
        .rst_n             (rst_n),

        .i_valid           (preq_valid),
        .o_ready           (rq_ready),

        .i_data            (preq_data),
        .i_rshift          (preq_rshift),

        .i_lane_valid_mask (preq_lane_valid_mask),
        .i_group_idx       (preq_group_idx),

        .o_valid           (o_valid),
        .i_out_ready       (i_out_ready),

        .o_data            (o_data),
        .o_lane_valid_mask (o_lane_valid_mask),
        .o_group_idx       (o_group_idx)
    );


endmodule