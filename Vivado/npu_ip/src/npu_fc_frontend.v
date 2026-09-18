`timescale 1ns / 1ps
`include "npu_defs.vh"

// Common tiled FC adapter for FC1 and FC2.
//
// The wrapper fixes FC_IN/FC_OUT/FC1_MODE from the manifest.  Keeping those
// values as elaboration constants deliberately avoids a START-edge divider or
// multiplier in the adapter.  One S_AXIS transfer is interpreted as:
//   activation -> all bias -> all rshift -> tile(group 0..last).
// Each tile is loaded only when B is ready to execute that group; B is stalled
// between groups while the next tile arrives.  Weight payload must already be
// padded and ordered as [group][chunk][lane][tap], INT16 little-endian.
module npu_fc_frontend #(
    parameter FC_IN       = 256,
    parameter FC_OUT      = 512,
    parameter FC1_MODE    = 1,
    parameter MAX_FC_IN   = 512,
    parameter MAX_FC_OUT  = 527,
    // Zero preserves the historical fixed-shape frontend for reduced tests.
    // The unified top sets this to one and supplies its latched descriptor.
    parameter USE_RUNTIME_DESC = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        i_start_pulse,
    input  wire        i_soft_reset_pulse,
    input  wire        i_desc_fc1_mode,
    input  wire [`NPU_CFG_DIM_W-1:0] i_desc_fc_in_dim,
    input  wire [`NPU_CFG_DIM_W-1:0] i_desc_fc_out_dim,
    output wire        o_busy,
    output wire        o_done,
    output wire        o_error_pulse,
    output wire [7:0]  o_error_code,

    output wire        s_axis_tready,
    input  wire [31:0] s_axis_tdata,
    input  wire [3:0]  s_axis_tkeep,
    input  wire        s_axis_tlast,
    input  wire        s_axis_tvalid,

    output wire        m_axis_tvalid,
    output wire [31:0] m_axis_tdata,
    output wire [3:0]  m_axis_tkeep,
    output wire        m_axis_tlast,
    input  wire        m_axis_tready,

    // ---- shared B-core boundary ----
    output wire        o_b_reset_request,
    output wire        o_b_start,
    output wire        o_b_fc1_mode,
    output wire [`NPU_CFG_DIM_W-1:0] o_b_fc_in_dim,
    output wire [`NPU_CFG_DIM_W-1:0] o_b_fc_out_dim,
    output wire        o_b_act_valid,
    input  wire        i_b_act_ready,
    output wire signed [15:0] o_b_x0,
    output wire signed [15:0] o_b_x1,
    output wire signed [15:0] o_b_x2,
    output wire        o_b_weight_valid,
    input  wire        i_b_weight_ready,
    output wire [`NPU_WEIGHT_BUS_W-1:0] o_b_weight,
    output wire [`NPU_BIAS_BUS_W-1:0]   o_b_bias,
    output wire [`NPU_RSHIFT_BUS_W-1:0] o_b_rshift,
    output wire                         o_b_rshift_valid,
    input  wire [`NPU_GROUP_W-1:0] i_b_result_group_idx,
    input  wire        i_b_out_valid,
    output wire        o_b_out_ready,
    input  wire [`NPU_OUT_BUS_W-1:0] i_b_out_data,
    input  wire [15:0] i_b_out_lane_mask,
    input  wire [`NPU_GROUP_W-1:0] i_b_out_group_idx
);

    localparam FC_CHUNKS     = (FC_IN  + 2) / 3;
    localparam FC_GROUPS     = (FC_OUT + 15) / 16;
    localparam FC_ACT_BEATS  = (FC_IN  + 1) / 2;
    localparam FC_BIAS_BEATS = FC_GROUPS * 32;  // 16 lanes * 48 bits
    localparam FC_RSH_BEATS  = FC_GROUPS * 4;   // 16 lanes * 8 bits
    localparam FC_TILE_BEATS = FC_CHUNKS * 24;  // 16 lanes * 3 taps / 2 words
    localparam MAX_FC_CHUNKS = (MAX_FC_IN + 2) / 3;
    localparam MAX_FC_GROUPS = (MAX_FC_OUT + 15) / 16;
    localparam CHUNK_W       = $clog2(MAX_FC_CHUNKS);
    localparam GROUP_W       = $clog2(MAX_FC_GROUPS);
    localparam [`NPU_CFG_DIM_W-1:0] FC_IN_CFG  = FC_IN;
    localparam [`NPU_CFG_DIM_W-1:0] FC_OUT_CFG = FC_OUT;

    // Runtime descriptor fields are captured at START.  The load/run FSM and
    // the B configuration then use only this operation-local copy; a later
    // CSR LAYER_ID write cannot become part of an active FC timing path.
    reg cfg_fc1_mode_reg;
    reg [`NPU_CFG_DIM_W-1:0] cfg_fc_in_dim_reg;
    reg [`NPU_CFG_DIM_W-1:0] cfg_fc_out_dim_reg;

    // FROZEN FC descriptors are represented as constants rather than a
    // runtime divider/multiplier.  In unified mode the selected descriptor is
    // latched at START and forwarded to B unchanged.
    wire cfg_fc1_mode = USE_RUNTIME_DESC ? cfg_fc1_mode_reg : (FC1_MODE != 0);
    wire [`NPU_CFG_DIM_W-1:0] cfg_fc_in_dim = USE_RUNTIME_DESC ? cfg_fc_in_dim_reg : FC_IN_CFG;
    wire [`NPU_CFG_DIM_W-1:0] cfg_fc_out_dim = USE_RUNTIME_DESC ? cfg_fc_out_dim_reg : FC_OUT_CFG;
    wire [CHUNK_W-1:0] cfg_fc_chunks = USE_RUNTIME_DESC ?
        (cfg_fc1_mode_reg ? 8'd86 : 8'd171) : FC_CHUNKS;
    wire [GROUP_W-1:0] cfg_fc_groups = USE_RUNTIME_DESC ?
        (cfg_fc1_mode_reg ? 6'd32 : 6'd33) : FC_GROUPS;
    wire [31:0] cfg_act_beats = USE_RUNTIME_DESC ?
        (cfg_fc1_mode_reg ? 32'd128 : 32'd256) : FC_ACT_BEATS;
    wire [31:0] cfg_tile_beats = USE_RUNTIME_DESC ?
        (cfg_fc1_mode_reg ? 32'd2064 : 32'd4104) : FC_TILE_BEATS;

    localparam [2:0] PH_LOAD_ACT            = 3'd0;
    localparam [2:0] PH_LOAD_BIAS           = 3'd1;
    localparam [2:0] PH_LOAD_RSHIFT         = 3'd2;
    localparam [2:0] PH_LOAD_TILE           = 3'd3;
    // The FC bias payload is held in 16 independent synchronous BRAM banks.
    // Each group is prefetched into the lane-local cache after its weight tile
    // arrives and before its first operand is issued.  Keeping this explicit
    // two-cycle phase makes the XPM read latency observable and prevents a
    // stale group from reaching B on a tile boundary.
    localparam [2:0] PH_PREFETCH_BIAS_REQ   = 3'd4;
    localparam [2:0] PH_PREFETCH_BIAS_LATCH = 3'd5;
    localparam [2:0] PH_RUN                 = 3'd6;

    localparam [7:0] ERR_TKEEP = 8'h30;
    localparam [7:0] ERR_TLAST = 8'h31;
    localparam [7:0] ERR_LEN   = 8'h32;

    reg op_active;
    reg [2:0] phase;
    reg b_start_pulse;
    reg b_rst_n_reg;

    wire adapter_clear = !rst_n || !b_rst_n_reg;
    wire start_accept = i_start_pulse && !op_active && !adapter_clear;

    assign o_busy = op_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            b_rst_n_reg <= 1'b0;
        else if (i_soft_reset_pulse || o_error_pulse)
            b_rst_n_reg <= 1'b0;
        else
            b_rst_n_reg <= 1'b1;
    end

    // Load counters and compact rotating decoders.  These are deliberately
    // separate from the XPM storage so reset clears only control/valid state.
    reg [31:0] load_beat;
    reg [CHUNK_W-1:0] act_chunk;
    reg [1:0] act_mod;
    reg [GROUP_W-1:0] bias_group;
    reg [3:0] bias_lane;
    reg bias_half;
    reg [31:0] bias_staging;
    reg [GROUP_W-1:0] rshift_group;
    reg [1:0] rshift_word;
    reg [GROUP_W-1:0] tile_group;
    // One FC chunk address fans out to all 48 lane/tap weight BRAMs.  Allow
    // synthesis to create physically local register replicas instead of
    // forcing one FF to span every BRAM column.  This is a placement/QoR
    // attribute only: the architectural state and cycle timing are unchanged.
    (* max_fanout = 4 *) reg [CHUNK_W-1:0] tile_chunk;
    reg [3:0] tile_lane;
    reg [1:0] tile_tap;

    // FC bias has 16 simultaneous lane reads while a group executes.  The
    // earlier flat register array cost 25,344 FF plus its static write/read
    // decode at the FC2 maximum.  Store one lane per XPM bank instead, then
    // prefetch the current group into the small lane-local cache below.
    //
    // Rshift stays in its existing register array for now: unlike bias, it is
    // consumed result-side against B's returned group index and must remain
    // combinationally result-aligned until that contract is separately piped.
    reg [31:0] rshift_mem_flat [0:MAX_FC_GROUPS*4-1];
    wire [47:0] bias_bank_rd [0:15];
    reg  [47:0] bias_cache_lane [0:15];
    reg         bias_cache_valid;

    // Store loader wires.
    wire [1:0] act_wr1_mod = (act_mod == 2'd2) ? 2'd0 : act_mod + 1'b1;
    wire [CHUNK_W-1:0] act_wr1_chunk = act_chunk + (act_mod == 2'd2);
    wire act_last_odd = cfg_fc_in_dim[0] && (load_beat == cfg_act_beats-1);

    // Advance a lane-major/tap-minor physical weight word position once.
    function [5:0] weight_pos_next;
        input [3:0] lane_in;
        input [1:0] tap_in;
        reg [3:0] lane_out;
        reg [1:0] tap_out;
        begin
            if (tap_in == 2'd2) begin
                tap_out = 2'd0;
                lane_out = lane_in + 1'b1;
            end else begin
                tap_out = tap_in + 1'b1;
                lane_out = lane_in;
            end
            weight_pos_next = {lane_out, tap_out};
        end
    endfunction

    wire [5:0] tile_pos1 = weight_pos_next(tile_lane, tile_tap);
    wire [3:0] tile_lane1 = tile_pos1[5:2];
    wire [1:0] tile_tap1  = tile_pos1[1:0];
    wire [5:0] tile_pos2 = weight_pos_next(tile_lane1, tile_tap1);
    wire [3:0] tile_lane2 = tile_pos2[5:2];
    wire [1:0] tile_tap2  = tile_pos2[1:0];

    // Only these four phases consume S_AXIS transfers.  The metadata-prefetch
    // states are internal bubbles; advertising TREADY there would consume the
    // first beat of the next tile before the cache is valid.
    wire is_load_phase = (phase == PH_LOAD_ACT) ||
                         (phase == PH_LOAD_BIAS) ||
                         (phase == PH_LOAD_RSHIFT) ||
                         (phase == PH_LOAD_TILE);
    wire s_axis_fire = s_axis_tvalid && s_axis_tready;
    wire final_stream_beat =
        (phase == PH_LOAD_TILE) &&
        (tile_group == cfg_fc_groups-1) &&
        (load_beat == cfg_tile_beats-1);
    wire tkeep_ok = (phase == PH_LOAD_ACT && act_last_odd) ?
                    (s_axis_tkeep == 4'h3) : (s_axis_tkeep == 4'hF);
    wire load_error = s_axis_fire && ((!tkeep_ok) || (s_axis_tlast != final_stream_beat));

    // Payload memories deliberately have no reset. Only controller/valid
    // state clears on abort, so a stale value can never be observed before a
    // complete new LOAD.
    // Decouple the live AXIS/load-counter cone from all 16 bias BRAM ports.
    // The second bias beat is accepted into this one-entry commit register;
    // the BRAM write occurs on the following clock from registered group/lane
    // and payload fields.  LOAD remains one beat/clock because capture and the
    // previous commit may overlap.
    wire bias_capture_fire = s_axis_fire && !load_error &&
                             (phase == PH_LOAD_BIAS) && bias_half;
    wire rshift_mem_wr = s_axis_fire && !load_error && (phase == PH_LOAD_RSHIFT);
    wire [GROUP_W+1:0] rshift_wr_addr = {rshift_group, 2'b00} + rshift_word;
    wire bias_prefetch_req = (phase == PH_PREFETCH_BIAS_REQ);

    reg                    bias_commit_valid;
    reg [GROUP_W-1:0]      bias_commit_group;
    reg [3:0]              bias_commit_lane;
    reg [47:0]             bias_commit_data;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_commit_valid <= 1'b0;
            bias_commit_group <= {GROUP_W{1'b0}};
            bias_commit_lane  <= 4'd0;
            bias_commit_data  <= 48'd0;
        end else if (adapter_clear || i_soft_reset_pulse || o_error_pulse) begin
            bias_commit_valid <= 1'b0;
        end else begin
            bias_commit_valid <= bias_capture_fire;
            if (bias_capture_fire) begin
                bias_commit_group <= bias_group;
                bias_commit_lane  <= bias_lane;
                bias_commit_data  <= {s_axis_tdata[15:0], bias_staging};
            end
        end
    end

    genvar bl;
    generate
        for (bl = 0; bl < 16; bl = bl + 1) begin : GEN_FC_BIAS_LANE
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A            (GROUP_W),
                .ADDR_WIDTH_B            (GROUP_W),
                .AUTO_SLEEP_TIME         (0),
                .BYTE_WRITE_WIDTH_A      (48),
                .CASCADE_HEIGHT          (0),
                .CLOCKING_MODE           ("common_clock"),
                .ECC_MODE                ("no_ecc"),
                .MEMORY_INIT_FILE        ("none"),
                .MEMORY_INIT_PARAM       ("0"),
                .MEMORY_OPTIMIZATION     ("true"),
                .MEMORY_PRIMITIVE        ("block"),
                .MEMORY_SIZE             (MAX_FC_GROUPS * 48),
                .MESSAGE_CONTROL         (0),
                .READ_DATA_WIDTH_B       (48),
                .READ_LATENCY_B          (1),
                .READ_RESET_VALUE_B      ("0"),
                .RST_MODE_A              ("SYNC"),
                .RST_MODE_B              ("SYNC"),
                .SIM_ASSERT_CHK          (0),
                .USE_EMBEDDED_CONSTRAINT (0),
                .USE_MEM_INIT            (0),
                .WAKEUP_TIME             ("disable_sleep"),
                .WRITE_DATA_WIDTH_A      (48),
                .WRITE_MODE_B            ("no_change")
            ) u_bias_bank (
                .clka           (clk), .ena(1'b1),
                .wea            (bias_commit_valid && (bias_commit_lane == bl[3:0])),
                .addra          (bias_commit_group),
                .dina           (bias_commit_data),
                .injectsbiterra (1'b0), .injectdbiterra(1'b0),
                .clkb           (clk), .rstb(1'b0), .enb(bias_prefetch_req),
                .regceb         (1'b1), .addrb(tile_group),
                .doutb          (bias_bank_rd[bl]), .sleep(1'b0),
                .dbiterrb       (), .sbiterrb()
            );
        end
    endgenerate

    genvar bg, rw;
    generate
        for (bg = 0; bg < MAX_FC_GROUPS; bg = bg + 1) begin : GEN_FC_RSHIFT_GROUP
            for (rw = 0; rw < 4; rw = rw + 1) begin : GEN_FC_RSHIFT_WORD
                always @(posedge clk) begin
                    if (rshift_mem_wr && (rshift_wr_addr == bg*4 + rw))
                        rshift_mem_flat[bg*4 + rw] <= s_axis_tdata;
                end
            end
        end
    endgenerate

    assign s_axis_tready = op_active && is_load_phase && !adapter_clear;
    assign o_error_pulse = op_active && load_error;
    assign o_error_code = !tkeep_ok ? ERR_TKEEP :
                          (s_axis_tlast != final_stream_beat) ? ERR_TLAST : ERR_LEN;

    wire store_act_wr0_en = s_axis_fire && !load_error && (phase == PH_LOAD_ACT);
    wire store_act_wr1_en = store_act_wr0_en && !act_last_odd;
    wire store_w_wr0_en = s_axis_fire && !load_error && (phase == PH_LOAD_TILE);

    // Capture a validated two-word FC weight transaction, then commit it to
    // the 48 BRAM banks one cycle later.  This removes load_beat/TLAST/TKEEP
    // validation from every BRAM EN cone while preserving one LOAD beat per
    // clock (capture and previous commit overlap).
    (* max_fanout = 4 *) reg weight_commit_valid;
    (* max_fanout = 4 *) reg [3:0]         weight_commit_lane0, weight_commit_lane1;
    (* max_fanout = 4 *) reg [1:0]         weight_commit_tap0,  weight_commit_tap1;
    (* max_fanout = 4 *) reg [CHUNK_W-1:0] weight_commit_chunk;
    (* max_fanout = 4 *) reg [15:0]        weight_commit_data0, weight_commit_data1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_commit_valid <= 1'b0;
            weight_commit_lane0 <= 4'd0;
            weight_commit_lane1 <= 4'd0;
            weight_commit_tap0  <= 2'd0;
            weight_commit_tap1  <= 2'd0;
            weight_commit_chunk <= {CHUNK_W{1'b0}};
            weight_commit_data0 <= 16'd0;
            weight_commit_data1 <= 16'd0;
        end else if (adapter_clear || !op_active || o_error_pulse) begin
            weight_commit_valid <= 1'b0;
        end else begin
            weight_commit_valid <= store_w_wr0_en;
            if (store_w_wr0_en) begin
                weight_commit_lane0 <= tile_lane;
                weight_commit_lane1 <= tile_lane1;
                weight_commit_tap0  <= tile_tap;
                weight_commit_tap1  <= tile_tap1;
                weight_commit_chunk <= tile_chunk;
                weight_commit_data0 <= s_axis_tdata[15:0];
                weight_commit_data1 <= s_axis_tdata[31:16];
            end
        end
    end

    // B operand issue: a single XPM read is held until both B input sides
    // accept it.  The read is one cycle ahead; never overwrite it while B
    // backpressures, so activation and weight remain paired and stable.
    reg operand_pending;
    // Same rationale as tile_chunk above.  The raw-route baseline showed
    // issue_chunk -> weight-BRAM address as the global worst path (fanout 59,
    // 94.7% route delay), so explicitly permit local FF replication.
    (* max_fanout = 4 *) reg [CHUNK_W-1:0] issue_chunk;
    reg all_groups_issued;
    wire fetch_req = (phase == PH_RUN) && bias_cache_valid && !operand_pending && !all_groups_issued;

    wire signed [15:0] store_x0, store_x1, store_x2;
    wire [`NPU_WEIGHT_BUS_W-1:0] store_weight;
    wire store_act_valid, store_weight_valid;
    wire operand_valid = operand_pending && store_act_valid && store_weight_valid;

    npu_fc_tile_store #(
        .MAX_FC_IN(MAX_FC_IN),
        .MAX_FC_CHUNKS(MAX_FC_CHUNKS)
    ) u_tile_store (
        .clk(clk), .rst_n(rst_n), .i_clear(adapter_clear || !op_active),
        .i_act_wr0_en(store_act_wr0_en), .i_act_wr0_bank(act_mod),
        .i_act_wr0_addr(act_chunk), .i_act_wr0_data(s_axis_tdata[15:0]),
        .i_act_wr1_en(store_act_wr1_en), .i_act_wr1_bank(act_wr1_mod),
        .i_act_wr1_addr(act_wr1_chunk), .i_act_wr1_data(s_axis_tdata[31:16]),
        .i_act_rd_en(fetch_req), .i_act_rd_addr(issue_chunk),
        .o_x0(store_x0), .o_x1(store_x1), .o_x2(store_x2), .o_act_rd_valid(store_act_valid),
        .i_w_wr0_en(weight_commit_valid), .i_w_wr0_lane(weight_commit_lane0), .i_w_wr0_tap(weight_commit_tap0),
        .i_w_wr0_addr(weight_commit_chunk), .i_w_wr0_data(weight_commit_data0),
        .i_w_wr1_en(weight_commit_valid), .i_w_wr1_lane(weight_commit_lane1), .i_w_wr1_tap(weight_commit_tap1),
        .i_w_wr1_addr(weight_commit_chunk), .i_w_wr1_data(weight_commit_data1),
        .i_w_rd_en(fetch_req), .i_w_rd_addr(issue_chunk), .o_weight(store_weight), .o_w_rd_valid(store_weight_valid)
    );

    // Issue-side bias must track the tile group.  Register B's result-side
    // group request locally before it selects the large rshift array.  When B
    // advances to a new group, o_b_rshift_valid inserts one boundary bubble
    // until this local address catches up; no result can consume stale shift.
    wire [`NPU_GROUP_W-1:0] result_group_idx;
    reg  [`NPU_GROUP_W-1:0] result_group_idx_reg;
    reg                     result_group_idx_valid;
    reg [`NPU_BIAS_BUS_W-1:0] selected_bias;
    reg [`NPU_RSHIFT_BUS_W-1:0] selected_rshift;
    integer sl;
    always @(*) begin
        selected_bias = {`NPU_BIAS_BUS_W{1'b0}};
        selected_rshift = {`NPU_RSHIFT_BUS_W{1'b0}};
        for (sl = 0; sl < 16; sl = sl + 1) begin
            selected_bias[sl*48 +: 48] = bias_cache_lane[sl];
            selected_rshift[sl*8 +: 8] = rshift_mem_flat[{result_group_idx, 2'b00} + (sl >> 2)][(sl & 3)*8 +: 8];
        end
    end

    wire b_act_ready = i_b_act_ready;
    wire b_weight_ready = i_b_weight_ready;
    wire b_fire = operand_valid && b_act_ready && b_weight_ready;

    // Keep RUN issue state in a physically small controller.  In particular,
    // load_beat/tile loader state no longer shares a priority mux with the
    // high-fanout issue_chunk address.  PH_RUN itself is the registered handoff
    // between the LOAD and RUN controllers.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            operand_pending  <= 1'b0;
            issue_chunk      <= {CHUNK_W{1'b0}};
            all_groups_issued <= 1'b0;
        end
        else if (adapter_clear || i_soft_reset_pulse || o_error_pulse || start_accept) begin
            operand_pending  <= 1'b0;
            issue_chunk      <= {CHUNK_W{1'b0}};
            all_groups_issued <= 1'b0;
        end
        else if (phase != PH_RUN) begin
            operand_pending  <= 1'b0;
            issue_chunk      <= {CHUNK_W{1'b0}};
            all_groups_issued <= 1'b0;
        end
        else begin
            if (fetch_req)
                operand_pending <= 1'b1;
            if (b_fire) begin
                operand_pending <= 1'b0;
                if (issue_chunk == cfg_fc_chunks-1) begin
                    issue_chunk <= {CHUNK_W{1'b0}};
                    if (tile_group == cfg_fc_groups-1)
                        all_groups_issued <= 1'b1;
                end
                else begin
                    issue_chunk <= issue_chunk + 1'b1;
                end
            end
        end
    end
    wire b_out_valid;
    wire [255:0] b_out_data;
    wire [15:0] b_out_mask;
    wire [`NPU_GROUP_W-1:0] b_out_group;
    wire b_out_ready;

    // Result serializer: one 256-bit FC group is drained as eight AXIS words.
    reg outbuf_valid;
    reg [255:0] outbuf_data;
    reg [15:0] outbuf_mask;
    reg [`NPU_GROUP_W-1:0] outbuf_group;
    reg [2:0] out_word;
    function [2:0] last_word_for_mask;
        input [15:0] mask;
        integer mw;
        begin
            last_word_for_mask = 3'd0;
            for (mw = 0; mw < 8; mw = mw + 1)
                if (mask[mw*2 +: 2] != 2'b00)
                    last_word_for_mask = mw[2:0];
        end
    endfunction
    wire [2:0] out_last_word = last_word_for_mask(outbuf_mask);
    wire [1:0] out_pair_mask = outbuf_mask[out_word*2 +: 2];
    wire m_axis_fire = m_axis_tvalid && m_axis_tready;

    assign b_out_ready = !outbuf_valid;
    assign m_axis_tvalid = outbuf_valid;
    assign m_axis_tdata = outbuf_data[out_word*32 +: 32];
    assign m_axis_tkeep = (out_pair_mask == 2'b11) ? 4'hF :
                          (out_pair_mask == 2'b01) ? 4'h3 :
                          (out_pair_mask == 2'b10) ? 4'hC : 4'h0;
    assign m_axis_tlast = outbuf_valid && (outbuf_group == cfg_fc_groups-1) && (out_word == out_last_word);
    assign o_done = m_axis_fire && m_axis_tlast;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_group_idx_reg   <= {`NPU_GROUP_W{1'b0}};
            result_group_idx_valid <= 1'b0;
        end
        else if (i_soft_reset_pulse) begin
            result_group_idx_reg   <= {`NPU_GROUP_W{1'b0}};
            result_group_idx_valid <= 1'b0;
        end
        else begin
            result_group_idx_reg   <= i_b_result_group_idx;
            result_group_idx_valid <= 1'b1;
        end
    end

    assign result_group_idx = result_group_idx_reg;
    assign b_out_valid = i_b_out_valid;
    assign b_out_data  = i_b_out_data;
    assign b_out_mask  = i_b_out_lane_mask;
    assign b_out_group = i_b_out_group_idx;

    assign o_b_reset_request = i_soft_reset_pulse || o_error_pulse;
    assign o_b_start         = b_start_pulse;
    assign o_b_fc1_mode      = cfg_fc1_mode;
    assign o_b_fc_in_dim     = cfg_fc_in_dim;
    assign o_b_fc_out_dim    = cfg_fc_out_dim;
    assign o_b_act_valid     = operand_valid;
    assign o_b_x0            = store_x0;
    assign o_b_x1            = store_x1;
    assign o_b_x2            = store_x2;
    assign o_b_weight_valid  = operand_valid;
    assign o_b_weight        = store_weight;
    assign o_b_bias          = selected_bias;
    assign o_b_rshift        = selected_rshift;
    assign o_b_rshift_valid  = result_group_idx_valid &&
                               (result_group_idx_reg == i_b_result_group_idx);
    assign o_b_out_ready     = b_out_ready;

    // Capture all 16 XPM read ports on the cycle after PH_PREFETCH_BIAS_REQ.
    // The cache payload itself deliberately has no reset: bias_cache_valid
    // below is the sole visibility guard, so abort/reset cannot expose stale
    // data while avoiding a wide reset fanout on these lane-local registers.
    integer bcl;
    always @(posedge clk) begin
        if (phase == PH_PREFETCH_BIAS_LATCH) begin
            for (bcl = 0; bcl < 16; bcl = bcl + 1)
                bias_cache_lane[bcl] <= bias_bank_rd[bcl];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_active <= 1'b0;
            phase <= PH_LOAD_ACT;
            b_start_pulse <= 1'b0;
            cfg_fc1_mode_reg <= 1'b0;
            cfg_fc_in_dim_reg <= FC_IN_CFG;
            cfg_fc_out_dim_reg <= FC_OUT_CFG;
            load_beat <= 0; act_chunk <= 0; act_mod <= 0;
            bias_group <= 0; bias_lane <= 0; bias_half <= 0; bias_staging <= 0;
            rshift_group <= 0; rshift_word <= 0;
            tile_group <= 0; tile_chunk <= 0; tile_lane <= 0; tile_tap <= 0;
            bias_cache_valid <= 1'b0;
            outbuf_valid <= 1'b0; outbuf_data <= 0; outbuf_mask <= 0; outbuf_group <= 0; out_word <= 0;
        end else if (adapter_clear || i_soft_reset_pulse || o_error_pulse) begin
            op_active <= 1'b0;
            phase <= PH_LOAD_ACT;
            b_start_pulse <= 1'b0;
            load_beat <= 0; act_chunk <= 0; act_mod <= 0;
            bias_group <= 0; bias_lane <= 0; bias_half <= 0; bias_staging <= 0;
            rshift_group <= 0; rshift_word <= 0;
            tile_group <= 0; tile_chunk <= 0; tile_lane <= 0; tile_tap <= 0;
            bias_cache_valid <= 1'b0;
            outbuf_valid <= 1'b0; out_word <= 0;
        end else begin
            b_start_pulse <= 1'b0;
            if (start_accept) begin
                op_active <= 1'b1;
                phase <= PH_LOAD_ACT;
                cfg_fc1_mode_reg <= i_desc_fc1_mode;
                cfg_fc_in_dim_reg <= i_desc_fc_in_dim;
                cfg_fc_out_dim_reg <= i_desc_fc_out_dim;
                load_beat <= 0; act_chunk <= 0; act_mod <= 0;
                bias_group <= 0; bias_lane <= 0; bias_half <= 0;
                rshift_group <= 0; rshift_word <= 0;
                tile_group <= 0; tile_chunk <= 0; tile_lane <= 0; tile_tap <= 0;
                bias_cache_valid <= 1'b0;
                outbuf_valid <= 1'b0; out_word <= 0;
            end

            if (s_axis_fire && !load_error) begin
                case (phase)
                    PH_LOAD_ACT: begin
                        if (load_beat == cfg_act_beats-1) begin
                            phase <= PH_LOAD_BIAS; load_beat <= 0;
                        end else begin
                            load_beat <= load_beat + 1'b1;
                        end
                        // Advance after the two sequential activation values.
                        if (act_mod == 2'd0) act_mod <= 2'd2;
                        else if (act_mod == 2'd1) begin act_mod <= 0; act_chunk <= act_chunk + 1'b1; end
                        else begin act_mod <= 2'd1; act_chunk <= act_chunk + 1'b1; end
                    end
                    PH_LOAD_BIAS: begin
                        load_beat <= load_beat + 1'b1;
                        if (!bias_half) begin
                            bias_staging <= s_axis_tdata;
                            bias_half <= 1'b1;
                        end else begin
                            bias_half <= 1'b0;
                            if (bias_lane == 4'd15) begin
                                bias_lane <= 0;
                                if (bias_group == cfg_fc_groups-1) begin
                                    phase <= PH_LOAD_RSHIFT; load_beat <= 0; rshift_group <= 0; rshift_word <= 0;
                                end else bias_group <= bias_group + 1'b1;
                            end else bias_lane <= bias_lane + 1'b1;
                        end
                    end
                    PH_LOAD_RSHIFT: begin
                        load_beat <= load_beat + 1'b1;
                        if (rshift_word == 2'd3) begin
                            rshift_word <= 0;
                            if (rshift_group == cfg_fc_groups-1) begin
                                phase <= PH_LOAD_TILE; load_beat <= 0; tile_chunk <= 0; tile_lane <= 0; tile_tap <= 0;
                            end else rshift_group <= rshift_group + 1'b1;
                        end else rshift_word <= rshift_word + 1'b1;
                    end
                    PH_LOAD_TILE: begin
                        load_beat <= load_beat + 1'b1;
                        tile_lane <= tile_lane2;
                        tile_tap <= tile_tap2;
                        if (tile_lane2 == 0 && tile_tap2 == 0) tile_chunk <= tile_chunk + 1'b1;
                        if (load_beat == cfg_tile_beats-1) begin
                            load_beat <= 0; tile_chunk <= 0; tile_lane <= 0; tile_tap <= 0;
                            phase <= PH_PREFETCH_BIAS_REQ;
                            bias_cache_valid <= 1'b0;
                            if (tile_group == 0) b_start_pulse <= 1'b1;
                        end
                    end
                    default: ;
                endcase
            end

            // Prefetch has no AXIS transfer.  It must therefore advance
            // independently of s_axis_fire; keeping it inside the LOAD case
            // would deadlock with TREADY correctly deasserted in these states.
            if (phase == PH_PREFETCH_BIAS_REQ) begin
                // XPM port-B address/en is sampled in this state.
                phase <= PH_PREFETCH_BIAS_LATCH;
            end
            else if (phase == PH_PREFETCH_BIAS_LATCH) begin
                // The separate cache always block captures the XPM output on
                // this edge; operands may be fetched from the next cycle.
                bias_cache_valid <= 1'b1;
                phase <= PH_RUN;
            end

            if (b_fire) begin
                if (issue_chunk == cfg_fc_chunks-1) begin
                    if (tile_group != cfg_fc_groups-1) begin
                        tile_group <= tile_group + 1'b1;
                        phase <= PH_LOAD_TILE;
                        load_beat <= 0; tile_chunk <= 0; tile_lane <= 0; tile_tap <= 0;
                    end
                end
            end

            if (b_out_valid && b_out_ready) begin
                outbuf_valid <= 1'b1;
                outbuf_data <= b_out_data;
                outbuf_mask <= b_out_mask;
                outbuf_group <= b_out_group;
                out_word <= 0;
            end else if (m_axis_fire) begin
                if (out_word == out_last_word) begin
                    outbuf_valid <= 1'b0;
                    out_word <= 0;
                end else out_word <= out_word + 1'b1;
            end

            if (o_done)
                op_active <= 1'b0;
        end
    end

endmodule
