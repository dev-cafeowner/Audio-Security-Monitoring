`timescale 1ns / 1ps

module maxpool1d48_16 #(
    parameter ACC_W      = 48,
    parameter SHIFT_W    = 8,
    parameter GROUP_W    = 4,
    parameter MAX_GROUPS = 16
)(
    input  wire clk,
    input  wire rst_n,

    // ============================================================
    // Input Stream
    // ============================================================
    input  wire                 i_valid,
    output wire                 o_ready,

    input  wire [16*ACC_W-1:0]  i_data,
    input  wire [16*SHIFT_W-1:0] i_rshift,
    input  wire [15:0]          i_lane_valid_mask,

    // Current output-channel group
    // 0 ~ 15 for max Cout=256
    input  wire [GROUP_W-1:0]   i_group_idx,

    // Current CONV time information
    //
    // i_time_mod3 = time_index % 3
    //
    // 0 : t = 0,3,6,...
    // 1 : t = 1,4,7,...
    // 2 : t = 2,5,8,...
    //
    // i_first_time distinguishes t=0 from t=3,6,...
    input  wire [1:0]           i_time_mod3,
    input  wire                 i_first_time,
    input  wire                 i_last_time,

    // ============================================================
    // Output Stream
    // ============================================================
    output reg                  o_valid,
    input  wire                 i_out_ready,

    output reg [16*ACC_W-1:0]   o_data,
    output reg [16*SHIFT_W-1:0] o_rshift,
    output reg [15:0]           o_lane_valid_mask,
    output reg [GROUP_W-1:0]    o_group_idx
);

    // ============================================================
    // Output Elastic Register
    // ============================================================
    assign o_ready =
        (~o_valid)
        |
        i_out_ready;


    wire input_fire;

    assign input_fire =
        i_valid
        &&
        o_ready;


    // ============================================================
    // Pool Window Rules
    //
    // PyTorch:
    // K=3, S=3, P=1
    //
    // Windows:
    //
    // p=0 : [-1, 0, 1]
    // p=1 : [ 2, 3, 4]
    // p=2 : [ 5, 6, 7]
    // ...
    //
    // ReLU is already applied, therefore padding value 0
    // cannot exceed a valid positive activation.
    // ============================================================

    // t=0 initializes the first window.
    //
    // t mod 3 = 2 initializes subsequent windows.
    wire init_window;

    assign init_window =
        i_first_time
        ||
        (i_time_mod3 == 2'd2);


    // t mod 3 = 0 (except t=0) is the second element
    // of a regular three-element window.
    wire update_window;

    assign update_window =
        (!i_first_time)
        &&
        (i_time_mod3 == 2'd0);


    // ============================================================
    // Output Generation
    //
    // Normal output:
    // t mod 3 = 1
    //
    // Examples:
    // t=1 -> max(x0,x1)
    // t=4 -> max(x2,x3,x4)
    //
    // Tail:
    // if last sample is phase 0:
    //
    // e.g. L=4
    // last t=3
    // max(x2,x3,0)
    //
    // Since ReLU output >=0:
    // max(x2,x3,0) == max(x2,x3)
    //
    // Special L=1:
    // max(0,x0,0) == x0
    // ============================================================
    wire emit_normal;
    wire emit_tail;
    wire emit_single;
    wire emit_now;

    assign emit_normal =
        (i_time_mod3 == 2'd1);

    assign emit_tail =
        (!i_first_time)
        &&
        i_last_time
        &&
        (i_time_mod3 == 2'd0);

    assign emit_single =
        i_first_time
        &&
        i_last_time;

    assign emit_now =
        emit_normal
        ||
        emit_tail
        ||
        emit_single;


    // ============================================================
    // Per-OC-Group Pool State
    //
    // One memory per lane:
    //
    // 16 lanes
    // × MAX_GROUPS
    // × 48 bits
    //
    // No reset is required for the data memories.
    // Every valid window overwrites its state before that
    // state is used.
    // ============================================================

    wire [16*ACC_W-1:0]
        pool_mem_read;

    wire [16*ACC_W-1:0]
        pool_mem_write;


    // Write when:
    //
    // 1) starting a new window
    // 2) processing the second value of a regular window
    //    unless it is the final sample, in which case the
    //    result can directly be emitted.
    wire pool_mem_we;

    assign pool_mem_we =
        input_fire
        &&
        (
            (
                init_window
                &&
                !emit_single
            )
            ||
            (
                update_window
                &&
                !emit_tail
            )
        );


    genvar g;

    generate

        for (
            g = 0;
            g < 16;
            g = g + 1
        ) begin : GEN_POOL_LANE

            reg signed [ACC_W-1:0]
                max_mem [0:MAX_GROUPS-1];


            wire signed [ACC_W-1:0]
                input_lane;

            wire signed [ACC_W-1:0]
                stored_lane;

            wire signed [ACC_W-1:0]
                max_lane;


            assign input_lane =
                $signed(
                    i_data[
                        g*ACC_W
                        +:
                        ACC_W
                    ]
                );


            assign stored_lane =
                max_mem[
                    i_group_idx
                ];


            assign max_lane =
                (
                    input_lane
                    >
                    stored_lane
                )
                ?
                input_lane
                :
                stored_lane;


            assign pool_mem_read[
                g*ACC_W
                +:
                ACC_W
            ] =
                stored_lane;


            // New window:
            //     max = current sample
            //
            // Window update:
            //     max = max(previous, current)
            assign pool_mem_write[
                g*ACC_W
                +:
                ACC_W
            ] =
                init_window
                ?
                input_lane
                :
                max_lane;


            always @(posedge clk) begin

                if (pool_mem_we) begin

                    max_mem[
                        i_group_idx
                    ] <=
                        pool_mem_write[
                            g*ACC_W
                            +:
                            ACC_W
                        ];

                end

            end

        end

    endgenerate


    // ============================================================
    // Output Generation
    // ============================================================
    integer lane;

    reg signed [ACC_W-1:0]
        input_lane_value;

    reg signed [ACC_W-1:0]
        stored_lane_value;


    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            o_valid <=
                1'b0;

            o_data <=
                {16*ACC_W{1'b0}};

            o_rshift <=
                {16*SHIFT_W{1'b0}};

            o_lane_valid_mask <=
                16'h0000;

            o_group_idx <=
                {GROUP_W{1'b0}};

        end
        else begin

            // ----------------------------------------------------
            // Consume previous output
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
            // Current input produces a pooled output
            // ----------------------------------------------------
            if (
                input_fire
                &&
                emit_now
            ) begin

                for (
                    lane = 0;
                    lane < 16;
                    lane = lane + 1
                ) begin

                    input_lane_value =
                        $signed(
                            i_data[
                                lane*ACC_W
                                +:
                                ACC_W
                            ]
                        );


                    stored_lane_value =
                        $signed(
                            pool_mem_read[
                                lane*ACC_W
                                +:
                                ACC_W
                            ]
                        );


                    // ============================================
                    // L=1 special case
                    //
                    // max(0, x0, 0) = x0
                    //
                    // Input is post-ReLU and therefore >= 0.
                    // ============================================
                    if (emit_single) begin

                        o_data[
                            lane*ACC_W
                            +:
                            ACC_W
                        ] <=
                            input_lane_value;

                    end

                    // ============================================
                    // Normal / tail window
                    // ============================================
                    else begin

                        if (
                            input_lane_value
                            >
                            stored_lane_value
                        ) begin

                            o_data[
                                lane*ACC_W
                                +:
                                ACC_W
                            ] <=
                                input_lane_value;

                        end
                        else begin

                            o_data[
                                lane*ACC_W
                                +:
                                ACC_W
                            ] <=
                                stored_lane_value;

                        end

                    end

                end


                o_lane_valid_mask <=
                    i_lane_valid_mask;

                o_group_idx <=
                    i_group_idx;

                o_rshift <=
                    i_rshift;

                o_valid <=
                    1'b1;

            end

        end

    end

endmodule
