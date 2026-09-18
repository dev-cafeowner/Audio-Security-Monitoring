`timescale 1ns / 1ps
`include "npu_defs.vh"

// FC activation/vector and current-output-group weight tile store.
//
// This is deliberately storage-only: frame counters, AXIS validation, B-core
// handshaking, bias/rshift storage, and output serialization belong to the
// parent npu_fc_backend_adapter. Keeping this module free of those concerns
// lets OOC synthesis measure the exact BRAM cost of the FC2 worst case.
//
// Activation is striped modulo 3. A chunk address therefore reads x0/x1/x2
// from three independent synchronous-read XPM banks in parallel. Weight is
// stored in the FINAL FROZEN lane-major/tap-minor physical layout using 48
// independent banks, one per B-core weight word.
module npu_fc_tile_store #(
    parameter MAX_FC_IN     = 512,
    parameter MAX_FC_CHUNKS = (MAX_FC_IN + 2) / 3
)(
    input  wire clk,
    input  wire rst_n,
    input  wire i_clear,

    // Activation loader: up to two adjacent INT16 values per AXIS beat.
    // Parent supplies the already-decoded modulo-3 bank/address; this module
    // intentionally contains no runtime divide/modulo arithmetic.
    input  wire                         i_act_wr0_en,
    input  wire [1:0]                   i_act_wr0_bank,
    input  wire [$clog2(MAX_FC_CHUNKS)-1:0] i_act_wr0_addr,
    input  wire signed [15:0]           i_act_wr0_data,
    input  wire                         i_act_wr1_en,
    input  wire [1:0]                   i_act_wr1_bank,
    input  wire [$clog2(MAX_FC_CHUNKS)-1:0] i_act_wr1_addr,
    input  wire signed [15:0]           i_act_wr1_data,

    // Read one FC chunk. Data is valid one clock after i_act_rd_en.
    input  wire                         i_act_rd_en,
    input  wire [$clog2(MAX_FC_CHUNKS)-1:0] i_act_rd_addr,
    output wire signed [15:0]           o_x0,
    output wire signed [15:0]           o_x1,
    output wire signed [15:0]           o_x2,
    output reg                          o_act_rd_valid,

    // Weight-tile loader: two physical words per AXIS beat. The parent must
    // guarantee that both slots never name the same lane/tap bank in one
    // cycle, exactly as the Conv 48-bank loader contract does.
    input  wire                         i_w_wr0_en,
    input  wire [3:0]                   i_w_wr0_lane,
    input  wire [1:0]                   i_w_wr0_tap,
    input  wire [$clog2(MAX_FC_CHUNKS)-1:0] i_w_wr0_addr,
    input  wire [15:0]                  i_w_wr0_data,
    input  wire                         i_w_wr1_en,
    input  wire [3:0]                   i_w_wr1_lane,
    input  wire [1:0]                   i_w_wr1_tap,
    input  wire [$clog2(MAX_FC_CHUNKS)-1:0] i_w_wr1_addr,
    input  wire [15:0]                  i_w_wr1_data,

    // Read the current FC chunk's complete 48-word B payload.
    input  wire                         i_w_rd_en,
    input  wire [$clog2(MAX_FC_CHUNKS)-1:0] i_w_rd_addr,
    output wire [`NPU_WEIGHT_BUS_W-1:0] o_weight,
    output reg                          o_w_rd_valid
);

    localparam FC_CHUNK_W = $clog2(MAX_FC_CHUNKS);

    // Valid flags, not payload memories, are reset. Stale payload is never
    // observable without the corresponding parent-level valid transaction.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || i_clear) begin
            o_act_rd_valid <= 1'b0;
            o_w_rd_valid   <= 1'b0;
        end else begin
            o_act_rd_valid <= i_act_rd_en;
            o_w_rd_valid   <= i_w_rd_en;
        end
    end

    wire signed [15:0] act_rd_data [0:2];

    genvar ab;
    generate
        for (ab = 0; ab < 3; ab = ab + 1) begin : GEN_ACT_BANK
            wire wr0_sel  = (i_act_wr0_bank == ab[1:0]);
            wire wr1_sel  = (i_act_wr1_bank == ab[1:0]);
            wire wr0_here = i_act_wr0_en && wr0_sel;
            wire wr1_here = i_act_wr1_en && wr1_sel;
            wire wr_here  = wr0_here || wr1_here;
            // Select payload/address from the decoded destination tag, not
            // from the error-gated write enable.  When wr_here=0 these inputs
            // are don't-care; keeping the load-error/final-frame cone off
            // BRAM D/ADDR avoids making validation logic a payload path.
            wire [FC_CHUNK_W-1:0] wr_addr = wr0_sel ? i_act_wr0_addr : i_act_wr1_addr;
            wire [15:0] wr_data = wr0_sel ? i_act_wr0_data : i_act_wr1_data;

            xpm_memory_sdpram #(
                .ADDR_WIDTH_A            (FC_CHUNK_W),
                .ADDR_WIDTH_B            (FC_CHUNK_W),
                .AUTO_SLEEP_TIME         (0),
                .BYTE_WRITE_WIDTH_A      (16),
                .CASCADE_HEIGHT          (0),
                .CLOCKING_MODE           ("common_clock"),
                .ECC_MODE                ("no_ecc"),
                .MEMORY_INIT_FILE        ("none"),
                .MEMORY_INIT_PARAM       ("0"),
                .MEMORY_OPTIMIZATION     ("true"),
                .MEMORY_PRIMITIVE        ("block"),
                .MEMORY_SIZE             (MAX_FC_CHUNKS * 16),
                .MESSAGE_CONTROL         (0),
                .READ_DATA_WIDTH_B       (16),
                .READ_LATENCY_B          (1),
                .READ_RESET_VALUE_B      ("0"),
                .RST_MODE_A              ("SYNC"),
                .RST_MODE_B              ("SYNC"),
                .SIM_ASSERT_CHK          (0),
                .USE_EMBEDDED_CONSTRAINT (0),
                .USE_MEM_INIT            (0),
                .WAKEUP_TIME             ("disable_sleep"),
                .WRITE_DATA_WIDTH_A      (16),
                .WRITE_MODE_B            ("no_change")
            ) u_act_bank (
                .clka           (clk), .ena(1'b1), .wea(wr_here),
                .addra          (wr_addr), .dina(wr_data),
                .injectsbiterra (1'b0), .injectdbiterra(1'b0),
                .clkb           (clk), .rstb(1'b0), .enb(i_act_rd_en),
                .regceb         (1'b1), .addrb(i_act_rd_addr),
                .doutb          (act_rd_data[ab]), .sleep(1'b0),
                .dbiterrb       (), .sbiterrb()
            );
        end
    endgenerate

    assign o_x0 = act_rd_data[0];
    assign o_x1 = act_rd_data[1];
    assign o_x2 = act_rd_data[2];

    wire [15:0] weight_rd_data [0:15][0:2];

    genvar wl, wt;
    generate
        for (wl = 0; wl < 16; wl = wl + 1) begin : GEN_WEIGHT_LANE
            for (wt = 0; wt < 3; wt = wt + 1) begin : GEN_WEIGHT_TAP
                wire wr0_sel  = (i_w_wr0_lane == wl[3:0]) &&
                                (i_w_wr0_tap == wt[1:0]);
                wire wr1_sel  = (i_w_wr1_lane == wl[3:0]) &&
                                (i_w_wr1_tap == wt[1:0]);
                wire wr0_here = i_w_wr0_en && wr0_sel;
                wire wr1_here = i_w_wr1_en && wr1_sel;
                wire wr_here  = wr0_here || wr1_here;
                wire [FC_CHUNK_W-1:0] wr_addr = wr0_sel ? i_w_wr0_addr : i_w_wr1_addr;
                wire [15:0] wr_data = wr0_sel ? i_w_wr0_data : i_w_wr1_data;

                xpm_memory_sdpram #(
                    .ADDR_WIDTH_A            (FC_CHUNK_W),
                    .ADDR_WIDTH_B            (FC_CHUNK_W),
                    .AUTO_SLEEP_TIME         (0),
                    .BYTE_WRITE_WIDTH_A      (16),
                    .CASCADE_HEIGHT          (0),
                    .CLOCKING_MODE           ("common_clock"),
                    .ECC_MODE                ("no_ecc"),
                    .MEMORY_INIT_FILE        ("none"),
                    .MEMORY_INIT_PARAM       ("0"),
                    .MEMORY_OPTIMIZATION     ("true"),
                    .MEMORY_PRIMITIVE        ("block"),
                    .MEMORY_SIZE             (MAX_FC_CHUNKS * 16),
                    .MESSAGE_CONTROL         (0),
                    .READ_DATA_WIDTH_B       (16),
                    .READ_LATENCY_B          (1),
                    .READ_RESET_VALUE_B      ("0"),
                    .RST_MODE_A              ("SYNC"),
                    .RST_MODE_B              ("SYNC"),
                    .SIM_ASSERT_CHK          (0),
                    .USE_EMBEDDED_CONSTRAINT (0),
                    .USE_MEM_INIT            (0),
                    .WAKEUP_TIME             ("disable_sleep"),
                    .WRITE_DATA_WIDTH_A      (16),
                    .WRITE_MODE_B            ("no_change")
                ) u_weight_bank (
                    .clka           (clk), .ena(1'b1), .wea(wr_here),
                    .addra          (wr_addr), .dina(wr_data),
                    .injectsbiterra (1'b0), .injectdbiterra(1'b0),
                    .clkb           (clk), .rstb(1'b0), .enb(i_w_rd_en),
                    .regceb         (1'b1), .addrb(i_w_rd_addr),
                    .doutb          (weight_rd_data[wl][wt]), .sleep(1'b0),
                    .dbiterrb       (), .sbiterrb()
                );

                assign o_weight[`NPU_WEIGHT_BIT_OFS(wl, wt) +: 16] =
                    weight_rd_data[wl][wt];
            end
        end
    endgenerate

endmodule
