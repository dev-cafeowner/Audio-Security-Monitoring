`timescale 1ns / 1ps

// v5 Pin4 lane: four adjacent input channels, three taps per channel.
// The registered multiplier/tap/reduction pipeline accepts one transaction
// every clock and produces one signed 36-bit partial sum per output lane.
module pin4_mac_lane_3tap #(
    parameter integer DATA_W   = 16,
    parameter integer WEIGHT_W = 16,
    parameter integer PIN      = 4,
    parameter integer TAPS     = 3
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         i_valid,
    input  wire [PIN*TAPS*DATA_W-1:0]   i_x,
    input  wire [PIN*TAPS*WEIGHT_W-1:0] i_weight,
    output wire [DATA_W+WEIGHT_W+4-1:0] o_sum
);
    localparam integer PROD_W    = DATA_W + WEIGHT_W;
    localparam integer TAP_SUM_W = PROD_W + 2;
    localparam integer SUM_W     = PROD_W + 4;

    reg signed [DATA_W-1:0]     x_s0 [0:PIN-1][0:TAPS-1];
    reg signed [WEIGHT_W-1:0]   w_s0 [0:PIN-1][0:TAPS-1];
    reg signed [PROD_W-1:0]     product_s1 [0:PIN-1][0:TAPS-1];
    reg signed [TAP_SUM_W-1:0]  tap_sum_s2 [0:PIN-1];
    reg signed [SUM_W-1:0]      total_sum_s3;

    reg valid_s0;
    reg valid_s1;
    reg valid_s2;
    integer cin_i;
    integer tap_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s0 <= 1'b0;
            valid_s1 <= 1'b0;
            valid_s2 <= 1'b0;
            total_sum_s3 <= {SUM_W{1'b0}};
            for (cin_i=0; cin_i<PIN; cin_i=cin_i+1) begin
                tap_sum_s2[cin_i] <= {TAP_SUM_W{1'b0}};
                for (tap_i=0; tap_i<TAPS; tap_i=tap_i+1) begin
                    x_s0[cin_i][tap_i] <= {DATA_W{1'b0}};
                    w_s0[cin_i][tap_i] <= {WEIGHT_W{1'b0}};
                    product_s1[cin_i][tap_i] <= {PROD_W{1'b0}};
                end
            end
        end else begin
            valid_s0 <= i_valid;
            valid_s1 <= valid_s0;
            valid_s2 <= valid_s1;

            if (i_valid) begin
                for (cin_i=0; cin_i<PIN; cin_i=cin_i+1)
                    for (tap_i=0; tap_i<TAPS; tap_i=tap_i+1) begin
                        x_s0[cin_i][tap_i] <=
                            i_x[((cin_i*TAPS+tap_i)*DATA_W) +: DATA_W];
                        w_s0[cin_i][tap_i] <=
                            i_weight[((cin_i*TAPS+tap_i)*WEIGHT_W) +: WEIGHT_W];
                    end
            end

            if (valid_s0)
                for (cin_i=0; cin_i<PIN; cin_i=cin_i+1)
                    for (tap_i=0; tap_i<TAPS; tap_i=tap_i+1)
                        product_s1[cin_i][tap_i] <=
                            x_s0[cin_i][tap_i] * w_s0[cin_i][tap_i];

            if (valid_s1)
                for (cin_i=0; cin_i<PIN; cin_i=cin_i+1)
                    tap_sum_s2[cin_i] <=
                        {{(TAP_SUM_W-PROD_W){product_s1[cin_i][0][PROD_W-1]}},product_s1[cin_i][0]} +
                        {{(TAP_SUM_W-PROD_W){product_s1[cin_i][1][PROD_W-1]}},product_s1[cin_i][1]} +
                        {{(TAP_SUM_W-PROD_W){product_s1[cin_i][2][PROD_W-1]}},product_s1[cin_i][2]};

            if (valid_s2)
                total_sum_s3 <=
                    {{(SUM_W-TAP_SUM_W){tap_sum_s2[0][TAP_SUM_W-1]}},tap_sum_s2[0]} +
                    {{(SUM_W-TAP_SUM_W){tap_sum_s2[1][TAP_SUM_W-1]}},tap_sum_s2[1]} +
                    {{(SUM_W-TAP_SUM_W){tap_sum_s2[2][TAP_SUM_W-1]}},tap_sum_s2[2]} +
                    {{(SUM_W-TAP_SUM_W){tap_sum_s2[3][TAP_SUM_W-1]}},tap_sum_s2[3]};
        end
    end

    assign o_sum = total_sum_s3;
endmodule

module mac_array_16x3x4 #(
    parameter integer DATA_W   = 16,
    parameter integer WEIGHT_W = 16,
    parameter integer LANES    = 16,
    parameter integer PIN      = 4,
    parameter integer TAPS     = 3
)(
    input  wire                                  clk,
    input  wire                                  rst_n,
    input  wire                                  i_valid,
    input  wire [PIN*TAPS*DATA_W-1:0]            i_x,
    input  wire [LANES*PIN*TAPS*WEIGHT_W-1:0]    i_weight,
    output wire                                  o_valid,
    output wire [LANES*(DATA_W+WEIGHT_W+4)-1:0]  o_mac_sum
);
    localparam integer LANE_WEIGHT_W = PIN*TAPS*WEIGHT_W;
    localparam integer SUM_W = DATA_W + WEIGHT_W + 4;
    reg valid_s0, valid_s1, valid_s2, valid_s3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s0 <= 1'b0;
            valid_s1 <= 1'b0;
            valid_s2 <= 1'b0;
            valid_s3 <= 1'b0;
        end else begin
            valid_s0 <= i_valid;
            valid_s1 <= valid_s0;
            valid_s2 <= valid_s1;
            valid_s3 <= valid_s2;
        end
    end

    genvar lane;
    generate
        for (lane=0; lane<LANES; lane=lane+1) begin : GEN_PIN4_LANE
            pin4_mac_lane_3tap #(
                .DATA_W(DATA_W),.WEIGHT_W(WEIGHT_W),.PIN(PIN),.TAPS(TAPS)
            ) u_lane (
                .clk(clk),.rst_n(rst_n),.i_valid(i_valid),.i_x(i_x),
                .i_weight(i_weight[lane*LANE_WEIGHT_W +: LANE_WEIGHT_W]),
                .o_sum(o_mac_sum[lane*SUM_W +: SUM_W])
            );
        end
    endgenerate

    assign o_valid = valid_s3;
endmodule
