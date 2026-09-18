`timescale 1ns / 1ps

module conv_postprocess_16 #(
    parameter ACC_W      = 48,
    parameter OUT_W      = 16,
    parameter SHIFT_W    = 8,
    parameter GROUP_W    = 4,
    parameter MAX_GROUPS = 16
)(
    input  wire clk,
    input  wire rst_n,

    // ============================================================
    // Input from Bias-aware Shared PE
    // ============================================================
    input  wire                    i_valid,
    output wire                    o_ready,

    input  wire [16*ACC_W-1:0]     i_acc,

    // Per-output-channel Rshift
    input  wire [16*SHIFT_W-1:0]   i_rshift,

    input  wire [15:0]             i_lane_valid_mask,
    input  wire [GROUP_W-1:0]      i_group_idx,

    // ============================================================
    // CONV Time Metadata
    // ============================================================
    input  wire [1:0]              i_time_mod3,
    input  wire                    i_first_time,
    input  wire                    i_last_time,

    // ============================================================
    // Layer Mode
    //
    // 0 : Conv1
    //     ReLU -> Requant
    //
    // 1 : Conv2~9
    //     ReLU -> MaxPool -> Requant
    // ============================================================
    input  wire                    i_pool_enable,

    // ============================================================
    // INT16 Output
    // ============================================================
    output wire                    o_valid,
    input  wire                    i_out_ready,

    output wire [16*OUT_W-1:0]     o_data,
    output wire [15:0]             o_lane_valid_mask,
    output wire [GROUP_W-1:0]      o_group_idx
);


    // ============================================================
    // ReLU
    // ============================================================
    wire relu_in_ready;

    wire relu_out_valid;
    wire relu_out_ready;

    wire [16*ACC_W-1:0]
        relu_out_data;

    wire [15:0]
        relu_out_lane_mask;


    assign o_ready =
        relu_in_ready;


    // ============================================================
    // Metadata traveling with ReLU output
    // ============================================================
    reg [16*SHIFT_W-1:0]
        relu_meta_rshift;

    reg [GROUP_W-1:0]
        relu_meta_group_idx;

    reg [1:0]
        relu_meta_time_mod3;

    reg
        relu_meta_first_time;

    reg
        relu_meta_last_time;

    reg
        relu_meta_pool_enable;


    wire relu_input_fire;

    assign relu_input_fire =
        i_valid
        &&
        relu_in_ready;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            relu_meta_rshift <=
                {16*SHIFT_W{1'b0}};

            relu_meta_group_idx <=
                {GROUP_W{1'b0}};

            relu_meta_time_mod3 <=
                2'd0;

            relu_meta_first_time <=
                1'b0;

            relu_meta_last_time <=
                1'b0;

            relu_meta_pool_enable <=
                1'b0;

        end
        else begin

            if (relu_input_fire) begin

                relu_meta_rshift <=
                    i_rshift;

                relu_meta_group_idx <=
                    i_group_idx;

                relu_meta_time_mod3 <=
                    i_time_mod3;

                relu_meta_first_time <=
                    i_first_time;

                relu_meta_last_time <=
                    i_last_time;

                relu_meta_pool_enable <=
                    i_pool_enable;

            end

        end

    end


    // ============================================================
    // ReLU Instance
    // ============================================================
    relu48_16 #(
        .ACC_W (ACC_W)
    ) u_relu48_16 (
        .clk               (clk),
        .rst_n             (rst_n),

        .i_valid           (i_valid),
        .o_ready           (relu_in_ready),

        .i_data            (i_acc),
        .i_lane_valid_mask (i_lane_valid_mask),

        .o_valid           (relu_out_valid),
        .i_out_ready       (relu_out_ready),

        .o_data            (relu_out_data),
        .o_lane_valid_mask (relu_out_lane_mask)
    );


    // ============================================================
    // MaxPool Path
    // ============================================================
    wire pool_in_valid;
    wire pool_in_ready;

    wire pool_out_valid;
    wire pool_out_ready;

    wire [16*ACC_W-1:0]
        pool_out_data;

    wire [15:0]
        pool_out_lane_mask;

    wire [GROUP_W-1:0]
        pool_out_group_idx;


    assign pool_in_valid =
        relu_out_valid
        &&
        relu_meta_pool_enable;


    // ============================================================
    // Pool Input Pipeline Register (poolin)
    //
    // Timing purpose:
    //
    // Breaks the long path:
    //
    // relu_meta_first_time_reg
    //   -> maxpool1d48_16 init_window/emit_single/emit_now decode
    //   -> 16-lane x 48b wide o_data select/write
    //
    // This stage DOES NOT change numeric operation ordering.
    // It is a plain 1-deep valid/ready register bundling data
    // and metadata together at the ReLU -> MaxPool module
    // boundary, so the wide o_data mux inside MaxPool is driven
    // from a register free to be placed near that mux, rather
    // than directly from relu_meta_first_time_reg.
    // ============================================================
    reg
        poolin_valid;

    reg [16*ACC_W-1:0]
        poolin_data;

    reg [16*SHIFT_W-1:0]
        poolin_rshift;

    reg [15:0]
        poolin_lane_mask;

    reg [GROUP_W-1:0]
        poolin_group_idx;

    reg [1:0]
        poolin_time_mod3;

    reg
        poolin_first_time;

    reg
        poolin_last_time;

    wire [16*SHIFT_W-1:0]
        pool_rshift;


    wire poolin_ready;

    assign poolin_ready =
        (~poolin_valid)
        |
        pool_in_ready;


    wire pool_input_fire;

    assign pool_input_fire =
        pool_in_valid
        &&
        poolin_ready;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            poolin_valid <=
                1'b0;

            poolin_data <=
                {16*ACC_W{1'b0}};

            poolin_rshift <=
                {16*SHIFT_W{1'b0}};

            poolin_lane_mask <=
                16'h0000;

            poolin_group_idx <=
                {GROUP_W{1'b0}};

            poolin_time_mod3 <=
                2'd0;

            poolin_first_time <=
                1'b0;

            poolin_last_time <=
                1'b0;

        end
        else begin

            if (poolin_ready) begin

                if (pool_input_fire) begin

                    poolin_valid <=
                        1'b1;

                    poolin_data <=
                        relu_out_data;

                    poolin_rshift <=
                        relu_meta_rshift;

                    poolin_lane_mask <=
                        relu_out_lane_mask;

                    poolin_group_idx <=
                        relu_meta_group_idx;

                    poolin_time_mod3 <=
                        relu_meta_time_mod3;

                    poolin_first_time <=
                        relu_meta_first_time;

                    poolin_last_time <=
                        relu_meta_last_time;

                end
                else begin

                    poolin_valid <=
                        1'b0;

                end

            end

        end

    end


    maxpool1d48_16 #(
        .ACC_W      (ACC_W),
        .SHIFT_W    (SHIFT_W),
        .GROUP_W    (GROUP_W),
        .MAX_GROUPS (MAX_GROUPS)
    ) u_maxpool1d48_16 (
        .clk               (clk),
        .rst_n             (rst_n),

        .i_valid           (poolin_valid),
        .o_ready           (pool_in_ready),

        .i_data            (poolin_data),
        .i_rshift          (poolin_rshift),
        .i_lane_valid_mask (poolin_lane_mask),

        .i_group_idx       (poolin_group_idx),

        .i_time_mod3       (poolin_time_mod3),
        .i_first_time      (poolin_first_time),
        .i_last_time       (poolin_last_time),

        .o_valid           (pool_out_valid),
        .i_out_ready       (pool_out_ready),

        .o_data            (pool_out_data),
        .o_rshift          (pool_rshift),
        .o_lane_valid_mask (pool_out_lane_mask),
        .o_group_idx       (pool_out_group_idx)
    );


    // ============================================================
    // Direct Conv1 Path
    // ============================================================
    wire direct_valid;
    wire direct_ready;


    assign direct_valid =
        relu_out_valid
        &&
        !relu_meta_pool_enable;


    // ============================================================
    // Source Arbitration
    //
    // Pool has priority because it may contain a previously
    // generated result while a new direct result is waiting.
    // ============================================================
    wire source_valid;
    wire source_ready;

    wire [16*ACC_W-1:0]
        source_data;

    wire [16*SHIFT_W-1:0]
        source_rshift;

    wire [15:0]
        source_lane_mask;

    wire [GROUP_W-1:0]
        source_group_idx;


    assign source_valid =
        pool_out_valid
        |
        direct_valid;


    assign source_data =
        pool_out_valid
        ?
        pool_out_data
        :
        relu_out_data;


    assign source_rshift =
        pool_out_valid
        ?
        pool_rshift
        :
        relu_meta_rshift;


    assign source_lane_mask =
        pool_out_valid
        ?
        pool_out_lane_mask
        :
        relu_out_lane_mask;


    assign source_group_idx =
        pool_out_valid
        ?
        pool_out_group_idx
        :
        relu_meta_group_idx;


    // ============================================================
    // Requant Input Pipeline Register
    //
    // Timing purpose:
    //
    // Breaks the long path:
    //
    // pool_group_idx
    //   -> rshift LUTRAM read
    //   -> variable requant logic
    //   -> INT16 output FF
    //
    // This stage DOES NOT change numeric operation ordering.
    // ============================================================
    reg rqbuf_valid;

    reg [ACC_W-1:0] rqbuf_data_lane [0:15];
    reg [SHIFT_W-1:0] rqbuf_rshift_lane [0:15];
    wire [16*ACC_W-1:0] rqbuf_data;
    wire [16*SHIFT_W-1:0] rqbuf_rshift;

    reg [15:0]
        rqbuf_lane_mask;

    reg [GROUP_W-1:0]
        rqbuf_group_idx;


    wire requant_in_ready;

    wire rqbuf_ready;


    assign rqbuf_ready =
        (~rqbuf_valid)
        |
        requant_in_ready;


    assign source_ready =
        rqbuf_ready;


    wire source_fire;

    assign source_fire =
        source_valid
        &&
        source_ready;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            rqbuf_valid <=
                1'b0;

            rqbuf_lane_mask <=
                16'h0000;

            rqbuf_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            if (rqbuf_ready) begin

                if (source_fire) begin

                    rqbuf_valid <=
                        1'b1;

                    rqbuf_lane_mask <=
                        source_lane_mask;

                    rqbuf_group_idx <=
                        source_group_idx;

                end
                else begin

                    rqbuf_valid <=
                        1'b0;

                end

            end

        end

    end

    // Lane-local payload registers preserve the exact rqbuf handshake while
    // allowing each 48-bit data/8-bit shift lane to sit beside its requant
    // consumer.  Total storage and latency are unchanged.
    genvar rq_lane;
    generate
        for (rq_lane = 0; rq_lane < 16; rq_lane = rq_lane + 1) begin : GEN_RQBUF_LANE
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rqbuf_data_lane[rq_lane]   <= {ACC_W{1'b0}};
                    rqbuf_rshift_lane[rq_lane] <= {SHIFT_W{1'b0}};
                end
                else if (rqbuf_ready && source_fire) begin
                    rqbuf_data_lane[rq_lane] <=
                        source_data[rq_lane*ACC_W +: ACC_W];
                    rqbuf_rshift_lane[rq_lane] <=
                        source_rshift[rq_lane*SHIFT_W +: SHIFT_W];
                end
            end

            assign rqbuf_data[rq_lane*ACC_W +: ACC_W] =
                rqbuf_data_lane[rq_lane];
            assign rqbuf_rshift[rq_lane*SHIFT_W +: SHIFT_W] =
                rqbuf_rshift_lane[rq_lane];
        end
    endgenerate


    // ============================================================
    // Source Ready Routing
    // ============================================================

    // Pool result gets arbitration priority.
    assign pool_out_ready =
        source_ready;


    // Direct result can advance only if no pool result is waiting.
    assign direct_ready =
        source_ready
        &&
        !pool_out_valid;


    // ReLU result either goes to Pool or direct arbitration path.
    assign relu_out_ready =
        relu_meta_pool_enable
        ?
        poolin_ready
        :
        direct_ready;


    // ============================================================
    // Requantization + INT16 Saturation
    //
    // Input now comes exclusively from registered rqbuf_* signals.
    // ============================================================
    requantize48_to_int16_16 #(
        .ACC_W   (ACC_W),
        .OUT_W   (OUT_W),
        .SHIFT_W (SHIFT_W),
        .GROUP_W (GROUP_W)
    ) u_requantize48_to_int16_16 (
        .clk               (clk),
        .rst_n             (rst_n),

        .i_valid           (rqbuf_valid),
        .o_ready           (requant_in_ready),

        .i_data            (rqbuf_data),
        .i_rshift          (rqbuf_rshift),

        .i_lane_valid_mask (rqbuf_lane_mask),
        .i_group_idx       (rqbuf_group_idx),

        .o_valid           (o_valid),
        .i_out_ready       (i_out_ready),

        .o_data            (o_data),
        .o_lane_valid_mask (o_lane_valid_mask),
        .o_group_idx       (o_group_idx)
    );

endmodule
