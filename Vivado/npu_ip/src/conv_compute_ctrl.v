`timescale 1ns / 1ps

module conv_compute_ctrl #(
    parameter CFG_C_W = 10,
    parameter CFG_T_W = 20
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 i_start,

    // One accepted MAC step
    input  wire                 i_step,

    // Layer configuration
    input  wire [CFG_C_W-1:0]   i_cin,
    input  wire [CFG_C_W-1:0]   i_cout,
    input  wire [CFG_T_W-1:0]   i_out_len,

    output wire                 o_compute_en,
    output wire                 o_clear,
    output wire                 o_last_cin,

    output wire [CFG_C_W-1:0]   o_cin_idx,
    output wire [CFG_C_W-1:0]   o_oc_group_idx,
    output wire [CFG_T_W-1:0]   o_time_idx,

    output reg                  o_busy,
    output reg                  o_done
);

    // ============================================================
    // Latched layer configuration
    // ============================================================
    reg [CFG_C_W-1:0] cin_cfg;
    reg [CFG_C_W-1:0] cout_cfg;
    reg [CFG_T_W-1:0] out_len_cfg;


    // ============================================================
    // Loop counters
    // ============================================================
    reg [CFG_C_W-1:0] cin_idx;
    reg [CFG_C_W-1:0] oc_group_idx;
    reg [CFG_T_W-1:0] time_idx;


    // ============================================================
    // Pout = 16
    // ============================================================
    wire [CFG_C_W-1:0] oc_group_count;
    wire [CFG_C_W-1:0] last_oc_group;

    assign oc_group_count =
        cout_cfg >> 4;

    assign last_oc_group =
        oc_group_count - 1'b1;


    // ============================================================
    // Current accepted compute cycle
    // ============================================================
    assign o_compute_en =
        o_busy && i_step;

    assign o_clear =
        o_busy &&
        i_step &&
        (cin_idx == {CFG_C_W{1'b0}});

    assign o_last_cin =
        o_busy &&
        i_step &&
        ((cin_idx + 4) >= cin_cfg);


    assign o_cin_idx      = cin_idx;
    assign o_oc_group_idx = oc_group_idx;
    assign o_time_idx     = time_idx;


    // ============================================================
    // Controller
    //
    // Loop order:
    //
    // Cin
    //   -> OC Group
    //      -> Time
    // ============================================================
    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            cin_cfg     <= {CFG_C_W{1'b0}};
            cout_cfg    <= {CFG_C_W{1'b0}};
            out_len_cfg <= {CFG_T_W{1'b0}};

            cin_idx      <= {CFG_C_W{1'b0}};
            oc_group_idx <= {CFG_C_W{1'b0}};
            time_idx     <= {CFG_T_W{1'b0}};

            o_busy <= 1'b0;
            o_done <= 1'b0;

        end
        else begin

            // done = 1-cycle pulse
            o_done <= 1'b0;


            // ====================================================
            // IDLE
            // ====================================================
            if (!o_busy) begin

                if (i_start) begin

                    cin_cfg     <= i_cin;
                    cout_cfg    <= i_cout;
                    out_len_cfg <= i_out_len;

                    cin_idx      <= {CFG_C_W{1'b0}};
                    oc_group_idx <= {CFG_C_W{1'b0}};
                    time_idx     <= {CFG_T_W{1'b0}};

                    o_busy <= 1'b1;

                end

            end


            // ====================================================
            // COMPUTE
            // ====================================================
            else begin

                // Only advance when one MAC step is accepted.
                if (i_step) begin

                    // --------------------------------------------
                    // Cin loop
                    // --------------------------------------------
                    if ((cin_idx + 4) >= cin_cfg) begin

                        cin_idx <=
                            {CFG_C_W{1'b0}};


                        // ----------------------------------------
                        // OC Group loop
                        // ----------------------------------------
                        if (oc_group_idx ==
                            last_oc_group) begin

                            oc_group_idx <=
                                {CFG_C_W{1'b0}};


                            // ------------------------------------
                            // Time loop
                            // ------------------------------------
                            if (time_idx ==
                                (out_len_cfg - 1'b1)) begin

                                time_idx <=
                                    {CFG_T_W{1'b0}};

                                o_busy <= 1'b0;
                                o_done <= 1'b1;

                            end
                            else begin

                                time_idx <=
                                    time_idx + 1'b1;

                            end

                        end
                        else begin

                            oc_group_idx <=
                                oc_group_idx + 1'b1;

                        end

                    end
                    else begin

                        cin_idx <=
                            cin_idx + 3'd4;

                    end

                end

            end

        end

    end

endmodule
