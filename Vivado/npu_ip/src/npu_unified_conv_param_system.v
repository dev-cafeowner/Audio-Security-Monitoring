`timescale 1ns / 1ps
`include "npu_defs.vh"

// One descriptor-driven parameter loader and one physical parameter store for
// every CONV layer (Conv1..Conv9). Conv1 is the Cin=1/group=4 member of the
// same canonical weight/bias/rshift stream contract used by Conv2..Conv9.
// RUN window generation remains frontend-specific.
module npu_unified_conv_param_system #(
    parameter integer MAX_CIN = 128,
    parameter integer MAX_OC_GROUP = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire i_start,
    input  wire i_flush,
    input  wire [`NPU_CFG_C_W-1:0] i_cin,
    input  wire [4:0] i_num_oc_group,

    output wire o_loading,
    output reg  o_params_ready_pulse,
    output reg  o_error_pulse,
    output reg  [7:0] o_error_code,

    output wire s_axis_tready,
    input  wire [31:0] s_axis_tdata,
    input  wire [3:0]  s_axis_tkeep,
    input  wire        s_axis_tlast,
    input  wire        s_axis_tvalid,

    input  wire [`NPU_CFG_C_W-1:0] i_issue_cin_idx,
    input  wire [`NPU_GROUP_W-1:0] i_issue_group_idx,
    input  wire [`NPU_GROUP_W-1:0] i_result_group_idx,
    output wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] o_weight,
    output wire [`NPU_BIAS_BUS_W-1:0]   o_bias,
    output wire [`NPU_BIAS_BUS_W-1:0]   o_bias_prefetch,
    output wire [`NPU_RSHIFT_BUS_W-1:0] o_rshift,
    output wire o_weight_valid,
    output wire o_bias_valid,
    output wire o_rshift_valid
);
    localparam [2:0] ST_IDLE    = 3'd0;
    localparam [2:0] ST_WEIGHT  = 3'd1;
    localparam [2:0] ST_BIAS    = 3'd2;
    localparam [2:0] ST_RSHIFT  = 3'd3;
    localparam [2:0] ST_DRAIN   = 3'd4;
    localparam [2:0] ST_PREFETCH= 3'd5;
    localparam [2:0] ST_READY   = 3'd6;

    localparam [7:0] ERR_LOAD_TKEEP  = 8'h10;
    localparam [7:0] ERR_LOAD_TLAST  = 8'h11;
    localparam [7:0] ERR_LOAD_LENGTH = 8'h12;
    localparam integer WBANK_DEPTH = MAX_OC_GROUP * MAX_CIN;
    localparam integer WBANK_AW = $clog2(WBANK_DEPTH);
    localparam integer WBANK_READ_DEPTH = WBANK_DEPTH / `NPU_PIN;
    localparam integer WBANK_READ_AW = $clog2(WBANK_READ_DEPTH);
    localparam integer GROUP_AW = (MAX_OC_GROUP <= 2) ? 1 : $clog2(MAX_OC_GROUP);

    reg [2:0] state;
    reg [`NPU_CFG_C_W-1:0] cin_last_r;
    reg [4:0] groups_r;
    reg [`NPU_CFG_C_W-1:0] last_weight_oc_r;
    reg [GROUP_AW-1:0] last_group_r;

    wire is_load = (state == ST_WEIGHT) || (state == ST_BIAS) || (state == ST_RSHIFT);
    wire axis_fire = s_axis_tvalid && s_axis_tready;

    assign o_loading = is_load || (state == ST_DRAIN) || (state == ST_PREFETCH);
    assign s_axis_tready = is_load && !i_flush;

    // Canonical K-fastest weight stream cursor. One 32-bit beat advances two
    // signed 16-bit elements and therefore produces two independent bank
    // write commands.
    reg [1:0] w_k_idx;
    reg [`NPU_CFG_C_W-1:0] w_ic_idx, w_oc_idx;
    function [2+2*`NPU_CFG_C_W-1:0] w_idx_next;
        input [1:0] k_in;
        input [`NPU_CFG_C_W-1:0] ic_in, oc_in, cin_last_in;
        reg [1:0] k_o;
        reg [`NPU_CFG_C_W-1:0] ic_o, oc_o;
        begin
            if (k_in == 2'd2) begin
                k_o = 2'd0;
                if (ic_in == cin_last_in) begin
                    ic_o = {`NPU_CFG_C_W{1'b0}};
                    oc_o = oc_in + 1'b1;
                end else begin
                    ic_o = ic_in + 1'b1;
                    oc_o = oc_in;
                end
            end else begin
                k_o = k_in + 1'b1;
                ic_o = ic_in;
                oc_o = oc_in;
            end
            w_idx_next = {k_o, ic_o, oc_o};
        end
    endfunction
    wire [2+2*`NPU_CFG_C_W-1:0] w_step1 = w_idx_next(w_k_idx,w_ic_idx,w_oc_idx,cin_last_r);
    wire [1:0] w_k_n1 = w_step1[2+2*`NPU_CFG_C_W-1 -: 2];
    wire [`NPU_CFG_C_W-1:0] w_ic_n1 = w_step1[2*`NPU_CFG_C_W-1 -: `NPU_CFG_C_W];
    wire [`NPU_CFG_C_W-1:0] w_oc_n1 = w_step1[`NPU_CFG_C_W-1:0];
    wire [2+2*`NPU_CFG_C_W-1:0] w_step2 = w_idx_next(w_k_n1,w_ic_n1,w_oc_n1,cin_last_r);

    reg weight_cmd_valid;
    reg [3:0] weight_cmd_lane0, weight_cmd_lane1;
    reg [1:0] weight_cmd_tap0, weight_cmd_tap1;
    reg [WBANK_AW-1:0] weight_cmd_addr0, weight_cmd_addr1;
    // Four physical payload clusters feed four adjacent lane groups.  The
    // copies are intentionally kept distinct: one global payload register
    // previously drove all 48 weight BRAMs and all 16 bias BRAMs, making the
    // BRAM D input route the raw-route critical cone.
    (* keep = "true" *) reg [15:0] weight_cmd_data0_cluster [0:3];
    (* keep = "true" *) reg [15:0] weight_cmd_data1_cluster [0:3];

    reg bias_half;
    reg [3:0] bias_lane;
    reg [GROUP_AW-1:0] bias_group_idx;
    reg [31:0] bias_staging;
    reg bias_cmd_valid;
    reg [3:0] bias_cmd_lane;
    reg [GROUP_AW-1:0] bias_cmd_group;
    (* keep = "true" *) reg [47:0] bias_cmd_data_cluster [0:3];

    reg [1:0] rshift_word;
    reg [GROUP_AW-1:0] rshift_group_idx;
    reg rshift_cmd_valid;
    reg [1:0] rshift_cmd_word;
    reg [GROUP_AW-1:0] rshift_cmd_group;
    reg [31:0] rshift_cmd_data;
    reg initial_prefetch_pulse;

    // Phase-local terminal predicates replace the former generic
    // beat_in_group/group_idx counter pair.  Each predicate is built only
    // from the cursor that already defines that phase's physical write
    // address, so the encoded loader state no longer drives a wide counter
    // CE network.
    wire weight_phase_last = (w_k_idx == 2'd1) &&
                             (w_ic_idx == cin_last_r) &&
                             (w_oc_idx == last_weight_oc_r);
    wire bias_phase_last = bias_half && (bias_lane == 4'd15) &&
                           (bias_group_idx == last_group_r);
    wire rshift_phase_last = (rshift_word == 2'd3) &&
                             (rshift_group_idx == last_group_r);
    wire phase_last = ((state == ST_WEIGHT) && weight_phase_last) ||
                      ((state == ST_BIAS) && bias_phase_last) ||
                      ((state == ST_RSHIFT) && rshift_phase_last);
    wire tkeep_bad = axis_fire && (s_axis_tkeep != 4'hF);
    wire tlast_early = axis_fire && s_axis_tlast && !phase_last;
    wire tlast_missing = axis_fire && phase_last && !s_axis_tlast;
    wire load_error = tkeep_bad || tlast_early || tlast_missing;
    wire [7:0] load_error_code = tkeep_bad ? ERR_LOAD_TKEEP :
                                 tlast_early ? ERR_LOAD_TLAST :
                                 tlast_missing ? ERR_LOAD_LENGTH : 8'h00;

    wire initial_store_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            cin_last_r <= 0; groups_r <= 0;
            last_weight_oc_r <= 0; last_group_r <= 0;
            w_k_idx <= 0; w_ic_idx <= 0; w_oc_idx <= 0;
            bias_half <= 0; bias_lane <= 0; bias_group_idx <= 0;
            bias_staging <= 0;
            rshift_word <= 0; rshift_group_idx <= 0;
            weight_cmd_valid <= 0; bias_cmd_valid <= 0; rshift_cmd_valid <= 0;
            initial_prefetch_pulse <= 0;
            o_params_ready_pulse <= 0;
            o_error_pulse <= 0;
            o_error_code <= 0;
        end else if (i_flush) begin
            state <= ST_IDLE;
            w_k_idx <= 0; w_ic_idx <= 0; w_oc_idx <= 0;
            bias_half <= 0; bias_lane <= 0; bias_group_idx <= 0;
            rshift_word <= 0; rshift_group_idx <= 0;
            weight_cmd_valid <= 0; bias_cmd_valid <= 0; rshift_cmd_valid <= 0;
            initial_prefetch_pulse <= 0;
            o_params_ready_pulse <= 0;
            o_error_pulse <= 0;
            o_error_code <= 0;
        end else begin
            weight_cmd_valid <= 1'b0;
            bias_cmd_valid <= 1'b0;
            rshift_cmd_valid <= 1'b0;
            initial_prefetch_pulse <= 1'b0;
            o_params_ready_pulse <= 1'b0;
            o_error_pulse <= 1'b0;

            if (load_error) begin
                o_error_pulse <= 1'b1;
                o_error_code <= load_error_code;
            end

            // AXI requires TDATA to remain stable while TVALID is asserted.
            // Capturing every valid cycle preserves the first bias word over
            // source bubbles while removing the loader phase decode from the
            // 32-bit staging register's CE network. On the second bias beat,
            // bias_cmd_data still observes the previous (low-word) value.
            if (s_axis_tvalid)
                bias_staging <= s_axis_tdata;
            // During all load phases TREADY is asserted continuously. Using
            // TVALID as the cursor advance removes the encoded phase decode
            // from the cursor CE. Cursor movement outside ST_WEIGHT is
            // unobservable because weight_cmd_valid is low and START resets
            // the cursor before the next weight frame.
            if (s_axis_tvalid)
                {w_k_idx,w_ic_idx,w_oc_idx} <= w_step2;

            if (i_start) begin
                state <= ST_WEIGHT;
                cin_last_r <= i_cin - 1'b1;
                groups_r <= i_num_oc_group;
                last_weight_oc_r <=
                    ({{(`NPU_CFG_C_W-5){1'b0}},i_num_oc_group} << 4) - 1'b1;
                last_group_r <= i_num_oc_group[GROUP_AW-1:0] - 1'b1;
                w_k_idx <= 0; w_ic_idx <= 0; w_oc_idx <= 0;
                bias_half <= 0; bias_lane <= 0; bias_group_idx <= 0;
                rshift_word <= 0; rshift_group_idx <= 0;
            end else if (axis_fire) begin
                if (load_error) begin
                    state <= ST_IDLE;
                end else begin
                    if (state == ST_WEIGHT) begin
                        weight_cmd_valid <= 1'b1;
                    end else if (state == ST_BIAS) begin
                        if (!bias_half) begin
                            bias_half <= 1'b1;
                        end else begin
                            bias_cmd_valid <= 1'b1;
                            bias_half <= 1'b0;
                            if (bias_lane == 4'd15) begin
                                bias_lane <= 4'd0;
                                bias_group_idx <= bias_group_idx + 1'b1;
                            end else begin
                                bias_lane <= bias_lane + 1'b1;
                            end
                        end
                    end else if (state == ST_RSHIFT) begin
                        rshift_cmd_valid <= 1'b1;
                        rshift_word <= rshift_word + 1'b1;
                        if (rshift_word == 2'd3)
                            rshift_group_idx <= rshift_group_idx + 1'b1;
                    end

                    if (phase_last) begin
                        bias_lane <= 0;
                        bias_half <= 0;
                        bias_group_idx <= 0;
                        rshift_word <= 0;
                        rshift_group_idx <= 0;
                        if (state == ST_WEIGHT) state <= ST_BIAS;
                        else if (state == ST_BIAS) state <= ST_RSHIFT;
                        else state <= ST_DRAIN;
                    end
                end
            end else if (state == ST_DRAIN) begin
                state <= ST_PREFETCH;
                initial_prefetch_pulse <= 1'b1;
            end else if ((state == ST_PREFETCH) && initial_store_ready) begin
                state <= ST_READY;
                o_params_ready_pulse <= 1'b1;
            end
        end
    end

    // Registered commit payloads are refreshed unconditionally; the
    // corresponding *_cmd_valid pulse is the sole write qualifier.  Keep
    // these no-reset payload registers out of the asynchronous-reset control
    // block above.  Leaving a register unassigned in an async-reset branch
    // can infer an LDCE feedback structure and place the reset release on the
    // weight-BRAM data path after SoC integration.
    always @(posedge clk) begin
        weight_cmd_lane0 <= w_oc_idx[3:0];
        weight_cmd_lane1 <= w_oc_n1[3:0];
        weight_cmd_tap0 <= w_k_idx;
        weight_cmd_tap1 <= w_k_n1;
        weight_cmd_addr0 <= {w_oc_idx[7:4],w_ic_idx[6:0]};
        weight_cmd_addr1 <= {w_oc_n1[7:4],w_ic_n1[6:0]};
        weight_cmd_data0_cluster[0] <= s_axis_tdata[15:0];
        weight_cmd_data0_cluster[1] <= s_axis_tdata[15:0];
        weight_cmd_data0_cluster[2] <= s_axis_tdata[15:0];
        weight_cmd_data0_cluster[3] <= s_axis_tdata[15:0];
        weight_cmd_data1_cluster[0] <= s_axis_tdata[31:16];
        weight_cmd_data1_cluster[1] <= s_axis_tdata[31:16];
        weight_cmd_data1_cluster[2] <= s_axis_tdata[31:16];
        weight_cmd_data1_cluster[3] <= s_axis_tdata[31:16];
        bias_cmd_lane <= bias_lane;
        bias_cmd_group <= bias_group_idx;
        bias_cmd_data_cluster[0] <= {s_axis_tdata[15:0],bias_staging};
        bias_cmd_data_cluster[1] <= {s_axis_tdata[15:0],bias_staging};
        bias_cmd_data_cluster[2] <= {s_axis_tdata[15:0],bias_staging};
        bias_cmd_data_cluster[3] <= {s_axis_tdata[15:0],bias_staging};
        rshift_cmd_word <= rshift_word;
        rshift_cmd_group <= rshift_group_idx;
        rshift_cmd_data <= s_axis_tdata;
    end

    // 48 weight banks, one per lane/tap.
    wire [63:0] wbank_rd [0:15][0:2];
    wire [WBANK_READ_AW-1:0] weight_rd_addr =
        {i_issue_group_idx[3:0],i_issue_cin_idx[6:2]};
    genvar wl, wt;
    generate
        for (wl=0; wl<16; wl=wl+1) begin : GEN_WBANK_LANE
            localparam integer WDATA_CLUSTER = wl / 4;
            for (wt=0; wt<3; wt=wt+1) begin : GEN_WBANK_TAP
                wire wr0 = weight_cmd_valid && (weight_cmd_lane0==wl[3:0]) && (weight_cmd_tap0==wt[1:0]);
                wire wr1 = weight_cmd_valid && (weight_cmd_lane1==wl[3:0]) && (weight_cmd_tap1==wt[1:0]);
                xpm_memory_sdpram #(
                    .ADDR_WIDTH_A(WBANK_AW),.ADDR_WIDTH_B(WBANK_READ_AW),.AUTO_SLEEP_TIME(0),
                    .BYTE_WRITE_WIDTH_A(16),.CASCADE_HEIGHT(0),.CLOCKING_MODE("common_clock"),
                    .ECC_MODE("no_ecc"),.MEMORY_INIT_FILE("none"),.MEMORY_INIT_PARAM("0"),
                    .MEMORY_OPTIMIZATION("true"),.MEMORY_PRIMITIVE("block"),
                    .MEMORY_SIZE(WBANK_DEPTH*16),.MESSAGE_CONTROL(0),
                    .READ_DATA_WIDTH_B(64),.READ_LATENCY_B(1),.READ_RESET_VALUE_B("0"),
                    .RST_MODE_A("SYNC"),.RST_MODE_B("SYNC"),.SIM_ASSERT_CHK(0),
                    .USE_EMBEDDED_CONSTRAINT(0),.USE_MEM_INIT(0),.WAKEUP_TIME("disable_sleep"),
                    .WRITE_DATA_WIDTH_A(16),.WRITE_MODE_B("no_change")
                ) u_wbank (
                    .clka(clk),.ena(1'b1),.wea(wr0|wr1),
                    .addra(wr0?weight_cmd_addr0:weight_cmd_addr1),
                    .dina(wr0 ? weight_cmd_data0_cluster[WDATA_CLUSTER]
                              : weight_cmd_data1_cluster[WDATA_CLUSTER]),
                    .injectsbiterra(1'b0),.injectdbiterra(1'b0),
                    .clkb(clk),.rstb(1'b0),.enb(1'b1),.regceb(1'b1),
                    .addrb(weight_rd_addr),.doutb(wbank_rd[wl][wt]),.sleep(1'b0),
                    .dbiterrb(),.sbiterrb()
                );
                assign o_weight[`NPU_PIN4_WEIGHT_BIT_OFS(wl,0,wt) +: 16] = wbank_rd[wl][wt][15:0];
                assign o_weight[`NPU_PIN4_WEIGHT_BIT_OFS(wl,1,wt) +: 16] = wbank_rd[wl][wt][31:16];
                assign o_weight[`NPU_PIN4_WEIGHT_BIT_OFS(wl,2,wt) +: 16] = wbank_rd[wl][wt][47:32];
                assign o_weight[`NPU_PIN4_WEIGHT_BIT_OFS(wl,3,wt) +: 16] = wbank_rd[wl][wt][63:48];
            end
        end
    endgenerate

    // Bias/rshift banks and group caches.
    wire [47:0] bias_bank_rd [0:15];
    wire [31:0] rshift_bank_rd [0:3];
    reg [47:0] bias_cache [0:15];
    reg [31:0] rshift_cache [0:3];
    reg [GROUP_AW-1:0] bias_cache_group, rshift_cache_group;
    reg bias_cache_valid, rshift_cache_valid;
    reg [GROUP_AW-1:0] bias_pending_group, rshift_pending_group;
    reg bias_pending, rshift_pending;
    wire [GROUP_AW-1:0] issue_group = i_issue_group_idx[GROUP_AW-1:0];
    wire [GROUP_AW-1:0] result_group = i_result_group_idx[GROUP_AW-1:0];
    wire bias_match = bias_cache_valid && (bias_cache_group==issue_group);
    wire rshift_match = rshift_cache_valid && (rshift_cache_group==result_group);
    wire bias_req = initial_prefetch_pulse || ((state==ST_READY) && !bias_match && !bias_pending);
    wire rshift_req = initial_prefetch_pulse || ((state==ST_READY) && !rshift_match && !rshift_pending);
    wire [GROUP_AW-1:0] bias_rd_addr = initial_prefetch_pulse ? 0 : issue_group;
    wire [GROUP_AW-1:0] rshift_rd_addr = initial_prefetch_pulse ? 0 : result_group;

    genvar bl, rw;
    generate
        for (bl=0; bl<16; bl=bl+1) begin : GEN_BIAS_LANE
            localparam integer BDATA_CLUSTER = bl / 4;
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A(GROUP_AW),.ADDR_WIDTH_B(GROUP_AW),.AUTO_SLEEP_TIME(0),
                .BYTE_WRITE_WIDTH_A(48),.CASCADE_HEIGHT(0),.CLOCKING_MODE("common_clock"),
                .ECC_MODE("no_ecc"),.MEMORY_INIT_FILE("none"),.MEMORY_INIT_PARAM("0"),
                .MEMORY_OPTIMIZATION("true"),.MEMORY_PRIMITIVE("block"),
                .MEMORY_SIZE(MAX_OC_GROUP*48),.MESSAGE_CONTROL(0),
                .READ_DATA_WIDTH_B(48),.READ_LATENCY_B(1),.READ_RESET_VALUE_B("0"),
                .RST_MODE_A("SYNC"),.RST_MODE_B("SYNC"),.SIM_ASSERT_CHK(0),
                .USE_EMBEDDED_CONSTRAINT(0),.USE_MEM_INIT(0),.WAKEUP_TIME("disable_sleep"),
                .WRITE_DATA_WIDTH_A(48),.WRITE_MODE_B("no_change")
            ) u_bias (
                .clka(clk),.ena(1'b1),.wea(bias_cmd_valid&&(bias_cmd_lane==bl[3:0])),
                .addra(bias_cmd_group),.dina(bias_cmd_data_cluster[BDATA_CLUSTER]),
                .injectsbiterra(1'b0),.injectdbiterra(1'b0),
                .clkb(clk),.rstb(1'b0),.enb(bias_req),.regceb(1'b1),
                .addrb(bias_rd_addr),.doutb(bias_bank_rd[bl]),.sleep(1'b0),
                .dbiterrb(),.sbiterrb()
            );
            assign o_bias[bl*48 +: 48] = bias_cache[bl];
            // ConvN's producer pipeline captures this synchronous BRAM
            // response one cycle after issuing its group address.  Keeping a
            // separate direct output lets Conv1 retain the legacy cache/valid
            // contract while eliminating ConvN group-boundary bubbles.
            assign o_bias_prefetch[bl*48 +: 48] = bias_bank_rd[bl];
        end
        for (rw=0; rw<4; rw=rw+1) begin : GEN_RSHIFT_WORD
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A(GROUP_AW),.ADDR_WIDTH_B(GROUP_AW),.AUTO_SLEEP_TIME(0),
                .BYTE_WRITE_WIDTH_A(32),.CASCADE_HEIGHT(0),.CLOCKING_MODE("common_clock"),
                .ECC_MODE("no_ecc"),.MEMORY_INIT_FILE("none"),.MEMORY_INIT_PARAM("0"),
                .MEMORY_OPTIMIZATION("true"),.MEMORY_PRIMITIVE("block"),
                .MEMORY_SIZE(MAX_OC_GROUP*32),.MESSAGE_CONTROL(0),
                .READ_DATA_WIDTH_B(32),.READ_LATENCY_B(1),.READ_RESET_VALUE_B("0"),
                .RST_MODE_A("SYNC"),.RST_MODE_B("SYNC"),.SIM_ASSERT_CHK(0),
                .USE_EMBEDDED_CONSTRAINT(0),.USE_MEM_INIT(0),.WAKEUP_TIME("disable_sleep"),
                .WRITE_DATA_WIDTH_A(32),.WRITE_MODE_B("no_change")
            ) u_rshift (
                .clka(clk),.ena(1'b1),.wea(rshift_cmd_valid&&(rshift_cmd_word==rw[1:0])),
                .addra(rshift_cmd_group),.dina(rshift_cmd_data),
                .injectsbiterra(1'b0),.injectdbiterra(1'b0),
                .clkb(clk),.rstb(1'b0),.enb(rshift_req),.regceb(1'b1),
                .addrb(rshift_rd_addr),.doutb(rshift_bank_rd[rw]),.sleep(1'b0),
                .dbiterrb(),.sbiterrb()
            );
            assign o_rshift[rw*32 +: 32] = rshift_cache[rw];
        end
    endgenerate

    integer ci;
    reg [WBANK_READ_AW-1:0] weight_addr_d1;
    reg weight_primed;
    always @(posedge clk) begin
        for (ci=0;ci<16;ci=ci+1) bias_cache[ci] <= bias_bank_rd[ci];
        for (ci=0;ci<4;ci=ci+1) rshift_cache[ci] <= rshift_bank_rd[ci];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || i_flush || i_start) begin
            bias_cache_valid<=0; rshift_cache_valid<=0;
            bias_pending<=0; rshift_pending<=0;
            bias_cache_group<=0; rshift_cache_group<=0;
            bias_pending_group<=0; rshift_pending_group<=0;
            weight_addr_d1<=0; weight_primed<=0;
        end else begin
            weight_addr_d1 <= weight_rd_addr;
            weight_primed <= 1'b1;
            if (bias_req) begin bias_cache_valid<=0; bias_pending<=1; bias_pending_group<=bias_rd_addr; end
            else if (bias_pending) begin bias_cache_valid<=1; bias_pending<=0; bias_cache_group<=bias_pending_group; end
            if (rshift_req) begin rshift_cache_valid<=0; rshift_pending<=1; rshift_pending_group<=rshift_rd_addr; end
            else if (rshift_pending) begin rshift_cache_valid<=1; rshift_pending<=0; rshift_cache_group<=rshift_pending_group; end
        end
    end

    assign o_weight_valid = (state==ST_READY) && weight_primed && (weight_rd_addr==weight_addr_d1);
    assign o_bias_valid = (state==ST_READY) && bias_match;
    assign o_rshift_valid = (state==ST_READY) && rshift_match;
    assign initial_store_ready = bias_cache_valid && (bias_cache_group==0) &&
                                 rshift_cache_valid && (rshift_cache_group==0) && weight_primed;
endmodule
