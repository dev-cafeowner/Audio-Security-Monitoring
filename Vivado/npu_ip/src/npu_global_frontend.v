`timescale 1ns / 1ps
`include "npu_defs.vh"

// GLOBAL frontend for neural_processing_unit_unified_2_0.
//
// This is the existing GLOBAL adapter's framing/assembly/drain logic with
// its private b_compute_top_16 instance removed.  The unified top owns the
// single B-core and connects this module through the explicit b_* boundary.
// No GLOBAL arithmetic, frame length, or AXIS packing rule is changed here.
module npu_global_frontend (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       i_start_pulse,
    input  wire                       i_soft_reset_pulse,
    output wire                       o_busy,
    output wire                       o_done,

    output wire                       s_axis_tready,
    input  wire [31:0]                s_axis_tdata,
    input  wire [3:0]                 s_axis_tkeep,
    input  wire                       s_axis_tlast,
    input  wire                       s_axis_tvalid,

    output wire                       m_axis_tvalid,
    output wire [31:0]                m_axis_tdata,
    output wire [3:0]                 m_axis_tkeep,
    output wire                       m_axis_tlast,
    input  wire                       m_axis_tready,

    // ---- shared B-core control / GLOBAL ingress ----
    output wire                       o_b_start,
    output wire                       o_b_global_valid,
    output wire [`NPU_GLOBAL_BUS_W-1:0] o_b_global_data,
    input  wire                       i_b_global_ready,

    // ---- shared B-core result / lifecycle ----
    input  wire                       i_b_out_valid,
    output wire                       o_b_out_ready,
    input  wire [`NPU_OUT_BUS_W-1:0]  i_b_out_data,
    input  wire [15:0]                i_b_out_lane_valid_mask,
    input  wire [`NPU_GROUP_W-1:0]    i_b_out_group_idx,
    input  wire                       i_b_busy,
    input  wire                       i_b_done
);

    // Input TKEEP/TLAST remain intentionally non-authoritative for GLOBAL.
    // They are retained as explicit ports to preserve the external AXIS shape.
    wire unused_s_axis_tkeep = &{1'b0, s_axis_tkeep};
    wire unused_s_axis_tlast = s_axis_tlast;
    wire unused_b_done = i_b_done;
    wire unused_lane_mask = &{1'b0, i_b_out_lane_valid_mask};

    wire frontend_clear = !rst_n || i_soft_reset_pulse;
    reg op_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            op_active <= 1'b0;
        else if (frontend_clear)
            op_active <= 1'b0;
        else if (i_start_pulse)
            op_active <= 1'b1;
        else if (o_done)
            op_active <= 1'b0;
    end

    // The unified top guarantees that i_start_pulse targets this frontend.
    assign o_b_start = i_start_pulse && !i_soft_reset_pulse;

    localparam ASM_COLLECT = 1'b0;
    localparam ASM_PRESENT = 1'b1;

    reg                         asm_state;
    reg [`NPU_GLOBAL_BUS_W-1:0] asm_shift_reg;
    reg [2:0]                   asm_word_idx;

    wire s_axis_fire = s_axis_tvalid && s_axis_tready;
    wire global_in_fire = o_b_global_valid && i_b_global_ready;

    assign s_axis_tready   = op_active && !i_soft_reset_pulse &&
                             (asm_state == ASM_COLLECT);
    assign o_b_global_valid = (asm_state == ASM_PRESENT);
    assign o_b_global_data  = asm_shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            asm_state     <= ASM_COLLECT;
            asm_word_idx  <= 3'd0;
            asm_shift_reg <= {`NPU_GLOBAL_BUS_W{1'b0}};
        end
        else if (frontend_clear) begin
            asm_state     <= ASM_COLLECT;
            asm_word_idx  <= 3'd0;
            asm_shift_reg <= {`NPU_GLOBAL_BUS_W{1'b0}};
        end
        else begin
            case (asm_state)
                ASM_COLLECT: if (s_axis_fire) begin
                    asm_shift_reg[asm_word_idx*32 +: 32] <= s_axis_tdata;
                    if (asm_word_idx == 3'd7) begin
                        asm_word_idx <= 3'd0;
                        asm_state    <= ASM_PRESENT;
                    end
                    else
                        asm_word_idx <= asm_word_idx + 3'd1;
                end
                ASM_PRESENT: if (global_in_fire)
                    asm_state <= ASM_COLLECT;
                default: asm_state <= ASM_COLLECT;
            endcase
        end
    end

    localparam DASM_IDLE  = 1'b0;
    localparam DASM_DRAIN = 1'b1;

    reg                      dasm_state;
    reg [`NPU_OUT_BUS_W-1:0] dasm_shift_reg;
    reg [2:0]                dasm_word_idx;
    reg                      dasm_is_last_group;

    wire core_out_fire = i_b_out_valid && o_b_out_ready;
    wire m_axis_fire   = m_axis_tvalid && m_axis_tready;

    assign o_b_out_ready = (dasm_state == DASM_IDLE);
    assign m_axis_tvalid = (dasm_state == DASM_DRAIN);
    assign m_axis_tdata  = dasm_shift_reg[dasm_word_idx*32 +: 32];
    assign m_axis_tkeep  = 4'hF;
    assign m_axis_tlast  = (dasm_state == DASM_DRAIN) &&
                           (dasm_word_idx == 3'd7) && dasm_is_last_group;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dasm_state        <= DASM_IDLE;
            dasm_shift_reg    <= {`NPU_OUT_BUS_W{1'b0}};
            dasm_word_idx     <= 3'd0;
            dasm_is_last_group <= 1'b0;
        end
        else if (frontend_clear) begin
            dasm_state        <= DASM_IDLE;
            dasm_shift_reg    <= {`NPU_OUT_BUS_W{1'b0}};
            dasm_word_idx     <= 3'd0;
            dasm_is_last_group <= 1'b0;
        end
        else begin
            case (dasm_state)
                DASM_IDLE: if (core_out_fire) begin
                    dasm_shift_reg     <= i_b_out_data;
                    dasm_is_last_group <= (i_b_out_group_idx == `NPU_GLOBAL_LAST_GROUP);
                    dasm_word_idx      <= 3'd0;
                    dasm_state         <= DASM_DRAIN;
                end
                DASM_DRAIN: if (m_axis_fire) begin
                    if (dasm_word_idx == 3'd7) begin
                        dasm_word_idx <= 3'd0;
                        dasm_state    <= DASM_IDLE;
                    end
                    else
                        dasm_word_idx <= dasm_word_idx + 3'd1;
                end
                default: dasm_state <= DASM_IDLE;
            endcase
        end
    end

    // `op_active` closes the start-to-first-B-busy gap; the drain state
    // keeps wrapper BUSY asserted until the final AXIS word is accepted.
    assign o_busy = op_active || i_b_busy || (dasm_state == DASM_DRAIN);
    assign o_done = m_axis_fire && m_axis_tlast;

endmodule
