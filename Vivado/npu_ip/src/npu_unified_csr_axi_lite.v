`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////
// npu_csr_axi_lite - AXI-Lite CSR implementing the LeeNet11 NPU wrapper
// register map (Docs/C_CSR_Register_Map_v1.md). Written as a reusable
// module: a backend engine (dummy or, later, the real B-core scheduler)
// connects through the o_start_pulse/o_soft_reset_pulse/i_busy/i_done
// handshake and the o_*_base_addr outputs, so this CSR can be reused
// unchanged when B is integrated - only the backend adapter changes.
//
// CONV1 SIBLING-IP COPY (Docs/C_Phase2_Conv1_LocalBuffer_Design_v1.md S8.2):
// this copy adds a backend error path (i_error_pulse/i_error_code) that the
// GLOBAL/dummy copies of this file do NOT have - npu_conv1_backend_adapter
// needs a way to report a malformed LOAD frame (bad TKEEP/TLAST) as a clean
// STATUS.ERROR fault instead of hanging forever. This is purely an
// additional backend-facing input port pair; the external AXI-Lite register
// map (0x00-0x1C) is byte-for-byte unchanged from the GLOBAL/dummy copies.
//
// C_S_AXI_ADDR_WIDTH=5 -> 8 x 32-bit registers (32B):
//   0x00 CTRL    - bit0 START (W1P), bit1 SOFT_RESET (W1P), bit2 IRQ_ENABLE (R/W)
//   0x04 STATUS  - bit0 BUSY (RO), bit1 DONE (sticky/W1C), bit2 ERROR (sticky/W1C),
//                  bits[15:8] ERROR_CODE (RO)
//   0x08 MODEL_BASE_ADDR  (R/W, 4-byte aligned)
//   0x0C INPUT_BASE_ADDR  (R/W, 4-byte aligned)
//   0x10 OUTPUT_BASE_ADDR (R/W, 4-byte aligned)
//   0x14 VERSION_ID (RO, fixed magic - matches SW/include/npu_regs.h NPU_VERSION_ID_MAGIC)
//   0x18 IRQ_STATUS - bit0 DONE_IRQ, bit1 ERROR_IRQ (RO/W1C; no physical irq
//                     output port yet - Docs/C_CSR_Register_Map_v1.md S6
//                     leaves IRQ usage undecided, and the current
//                     validation target is polling via npu_wait_done(), so
//                     only the register bits are implemented here)
//   0x1C DUMMY_MODE - backend-specific test-mode selector for THIS dummy
//                     build only. SW/include/npu_regs.h keeps this offset
//                     as RESERVED_1C in the common/reusable ABI - DUMMY_MODE
//                     is a name used only in this file's comments/dummy
//                     driver test code, not part of the final npu_wrapper ABI.
//
// Recommended driver sequence once this CSR is wired in:
//   mode/base address setup -> arm S2MM -> write CTRL.START -> start MM2S
// (S2MM-before-MM2S per Docs/C_DMA_Loopback_Regression_v1.md S1; CTRL.START
// before MM2S start because the backend's s_axis_tready/BUSY gate only
// opens after an accepted START.)
//
// Priority rules:
// - START/SOFT_RESET same-write: if a single CTRL write sets both bits,
//   SOFT_RESET wins and no start pulse is generated that cycle.
// - STATUS W1C vs new event: if a W1C write to DONE/ERROR lands the same
//   cycle a new DONE/ERROR event arrives, the new event wins (SET beats
//   CLEAR) - a completion/error can never be silently lost to a same-cycle
//   clear. Same rule applies to IRQ_STATUS.
// - CFG writes while BUSY: a write to MODEL/INPUT/OUTPUT_BASE_ADDR while
//   BUSY is ignored and latches STATUS.ERROR with
//   ERROR_CODE=CFG_WRITE_WHILE_BUSY (0x02), distinct from
//   ERROR_CODE=ALIGNMENT (0x01) for a non-4-byte-aligned write while idle.
// - Auto-clear on START (Docs/C_CSR_Register_Map_v1.md S3): an accepted
//   START clears STATUS.DONE and IRQ_STATUS.DONE_IRQ, so a fresh operation
//   never starts under a stale DONE left over from the previous run.
//   STATUS.ERROR/ERROR_CODE and IRQ_STATUS.ERROR_IRQ are NOT cleared by
//   START - they are diagnostic information that must survive until
//   explicitly W1C'd or SOFT_RESET, not be silently wiped by the next run.
//////////////////////////////////////////////////////////////////////////////

module npu_unified_csr_axi_lite #
(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 5
)
(
    // ---- Backend interface ----
    output wire        o_start_pulse,
    output wire        o_soft_reset_pulse,
    input  wire        i_busy,
    input  wire        i_done,
    input  wire        i_error_pulse,
    input  wire [7:0]  i_error_code,
    output wire [3:0]  o_layer_id,
    output wire [31:0] o_model_base_addr,
    output wire [31:0] o_input_base_addr,
    output wire [31:0] o_output_base_addr,

    // ---- AXI-Lite slave ----
    input  wire  S_AXI_ACLK,
    input  wire  S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
    input  wire [2 : 0] S_AXI_AWPROT,
    input  wire  S_AXI_AWVALID,
    output wire  S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
    input  wire  S_AXI_WVALID,
    output wire  S_AXI_WREADY,
    output wire [1 : 0] S_AXI_BRESP,
    output wire  S_AXI_BVALID,
    input  wire  S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
    input  wire [2 : 0] S_AXI_ARPROT,
    input  wire  S_AXI_ARVALID,
    output wire  S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
    output wire [1 : 0] S_AXI_RRESP,
    output wire  S_AXI_RVALID,
    input  wire  S_AXI_RREADY
);

    localparam ADDR_LSB = 2;  // 32-bit registers

    localparam [2:0] REG_CTRL      = 3'h0; // 0x00
    localparam [2:0] REG_STATUS    = 3'h1; // 0x04
    localparam [2:0] REG_MODEL     = 3'h2; // 0x08
    localparam [2:0] REG_INPUT     = 3'h3; // 0x0C
    localparam [2:0] REG_OUTPUT    = 3'h4; // 0x10
    localparam [2:0] REG_VERSION   = 3'h5; // 0x14
    localparam [2:0] REG_IRQ       = 3'h6; // 0x18
    localparam [2:0] REG_LAYER_ID  = 3'h7; // 0x1C

    localparam [31:0] VERSION_ID_MAGIC = 32'h4E505531; // "NPU1", matches SW/include/npu_regs.h

    localparam [7:0] ERR_NONE                = 8'h00;
    localparam [7:0] ERR_ALIGNMENT           = 8'h01;
    localparam [7:0] ERR_CFG_WRITE_WHILE_BUSY = 8'h02;
    localparam [7:0] ERR_INVALID_LAYER_ID    = 8'h16;

    // Declared early (ahead of its first use in o_start_pulse/cfg_write_while_busy
    // below) - xvlog requires reg/wire declarations to textually precede use.
    reg busy_reg;
    reg [3:0] layer_id_reg;

    // ============================================================
    // AXI-Lite write/read channel handshake - same skeleton as the
    // Vivado-wizard AXI-Lite slave this replaces (dummy_npu_slave_lite_v1_0_S00_AXI.v).
    // ============================================================
    reg axi_awready, axi_wready, axi_bvalid;
    reg [1:0] axi_bresp;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;

    reg axi_arready, axi_rvalid;
    reg [1:0] axi_rresp;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    // ============================================================
    // Write FSM: strict three-phase AW -> W -> B, unlike the wizard
    // template this replaced (which kept WREADY high through Waddr and
    // could lose a W that arrived before AW - a legal AXI4-Lite master
    // ordering). AWREADY is only asserted while waiting for AW (Waddr),
    // WREADY only while waiting for W (Wdata, i.e. AW already latched) - a
    // master sending W early simply sees WREADY=0 and, per AXI4-Lite, must
    // hold WVALID/WDATA/WSTRB stable until this FSM reaches Wdata - so no
    // write can ever be silently dropped or matched to a stale address.
    // A dedicated Bresp state (rather than re-asserting AWREADY the same
    // cycle BVALID is set) guarantees exactly one B response per accepted
    // write: AWREADY only comes back once BREADY has consumed this BVALID,
    // so a second AW/W can never land before the first write's response
    // has been collected and get coalesced into it.
    // ============================================================
    reg [1:0] state_write;
    reg [1:0] state_read;
    localparam Idle = 2'b00, Waddr = 2'b10, Wdata = 2'b11, Bresp = 2'b01,
               Raddr = 2'b10, Rdata = 2'b11;

    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            axi_awready <= 0;
            axi_wready  <= 0;
            axi_bvalid  <= 0;
            axi_bresp   <= 0;
            axi_awaddr  <= 0;
            state_write <= Idle;
        end
        else begin
            case (state_write)
                Idle: begin
                    if (S_AXI_ARESETN == 1'b1) begin
                        axi_awready <= 1'b1;
                        axi_wready  <= 1'b0;
                        state_write <= Waddr;
                    end
                end
                Waddr: begin
                    if (S_AXI_AWVALID && S_AXI_AWREADY) begin
                        axi_awaddr  <= S_AXI_AWADDR;
                        axi_awready <= 1'b0;
                        axi_wready  <= 1'b1;
                        state_write <= Wdata;
                    end
                end
                Wdata: begin
                    if (S_AXI_WVALID && S_AXI_WREADY) begin
                        axi_wready <= 1'b0;
                        axi_bvalid <= 1'b1;
                        state_write <= Bresp;
                    end
                end
                Bresp: begin
                    if (S_AXI_BREADY && axi_bvalid) begin
                        axi_bvalid  <= 1'b0;
                        axi_awready <= 1'b1;
                        state_write <= Waddr;
                    end
                end
                default: state_write <= Idle;
            endcase
        end
    end

    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
            axi_rresp   <= 1'b0;
            state_read  <= Idle;
        end
        else begin
            case (state_read)
                Idle: begin
                    if (S_AXI_ARESETN == 1'b1) begin
                        state_read  <= Raddr;
                        axi_arready <= 1'b1;
                    end
                end
                Raddr: begin
                    if (S_AXI_ARVALID && S_AXI_ARREADY) begin
                        state_read  <= Rdata;
                        axi_araddr  <= S_AXI_ARADDR;
                        axi_rvalid  <= 1'b1;
                        axi_arready <= 1'b0;
                    end
                end
                Rdata: begin
                    if (S_AXI_RVALID && S_AXI_RREADY) begin
                        axi_rvalid  <= 1'b0;
                        axi_arready <= 1'b1;
                        state_read  <= Raddr;
                    end
                end
                default: state_read <= Idle;
            endcase
        end
    end

    // ============================================================
    // write_commit: the one cycle a write's address and data are both
    // valid and correctly paired. With the strict two-phase FSM above,
    // there is exactly one commit point - W is only ever accepted (WREADY=1)
    // in Wdata, which is only reached after AW has already been latched
    // into axi_awaddr - so this always uses that latched address, never a
    // stale one from a prior transaction.
    // ============================================================
    wire write_commit = (state_write == Wdata) && S_AXI_WVALID && S_AXI_WREADY;
    wire [2:0] write_commit_sel = axi_awaddr[ADDR_LSB+2:ADDR_LSB];

    // S_AXI_WDATA/S_AXI_WSTRB are valid and stable on the write_commit cycle
    // (AXI4-Lite requires the master to hold them through WVALID&&!WREADY,
    // and both commit cases above require WVALID=1), so they can be used
    // directly wherever write_commit is true.
    wire wr_ctrl        = write_commit && (write_commit_sel == REG_CTRL)   && S_AXI_WSTRB[0];
    wire wr_status       = write_commit && (write_commit_sel == REG_STATUS) && S_AXI_WSTRB[0];
    wire wr_model         = write_commit && (write_commit_sel == REG_MODEL);
    wire wr_input        = write_commit && (write_commit_sel == REG_INPUT);
    wire wr_output       = write_commit && (write_commit_sel == REG_OUTPUT);
    wire wr_irq           = write_commit && (write_commit_sel == REG_IRQ)     && S_AXI_WSTRB[0];
    wire wr_layer_id    = write_commit && (write_commit_sel == REG_LAYER_ID) && S_AXI_WSTRB[0];
    wire wr_any_base    = wr_model || wr_input || wr_output;
    wire wr_any_config  = wr_any_base || wr_layer_id;

    wire [2:0] rd_sel = axi_araddr[ADDR_LSB+2:ADDR_LSB];

    // ============================================================
    // CTRL (0x00): START/SOFT_RESET are W1P pulses, never stored.
    // SOFT_RESET wins if both bits are set in the same write. START is
    // additionally gated on !busy_reg - the backend contract requires
    // start only while busy=0 (Docs/C_CSR_Register_Map_v1.md S2/S3); the
    // current dummy backend happens to ignore a stray start-while-busy
    // itself, but this CSR is meant to be reused as-is by a real backend
    // that may not be as forgiving.
    // ============================================================
    wire ctrl_wr_start_bit      = S_AXI_WDATA[0];
    wire ctrl_wr_soft_reset_bit = S_AXI_WDATA[1];
    wire ctrl_wr_irq_enable_bit = S_AXI_WDATA[2];

    wire layer_id_valid = (layer_id_reg < 4'd12);
    wire invalid_start = wr_ctrl && ctrl_wr_start_bit && !ctrl_wr_soft_reset_bit &&
                         !busy_reg && !layer_id_valid;
    assign o_start_pulse      = wr_ctrl && ctrl_wr_start_bit && !ctrl_wr_soft_reset_bit &&
                                !busy_reg && layer_id_valid;
    assign o_soft_reset_pulse = wr_ctrl && ctrl_wr_soft_reset_bit;

    reg irq_enable_reg;

    // ============================================================
    // Backend config registers.
    // MODEL/INPUT/OUTPUT_BASE_ADDR: rejected (write ignored, error latched)
    // while BUSY, and rejected (write ignored, error latched) if not
    // 4-byte aligned (evaluated only when byte 0 - which holds bits[1:0] -
    // is actually part of this write, i.e. S_AXI_WSTRB[0]). Accepted writes
    // are applied per-byte per S_AXI_WSTRB, matching the wizard template's
    // own byte-enable convention. DUMMY_MODE is a plain R/W test-selector
    // with neither restriction - it's not a DDR pointer.
    // ============================================================
    reg [31:0] model_base_addr_reg;
    reg [31:0] input_base_addr_reg;
    reg [31:0] output_base_addr_reg;
    wire cfg_write_while_busy = wr_any_config && busy_reg && (|S_AXI_WSTRB);
    wire addr_misaligned      = wr_any_base && !busy_reg && S_AXI_WSTRB[0] && (S_AXI_WDATA[1:0] != 2'b00);
    wire accept_base_write    = wr_any_base && !busy_reg && !(S_AXI_WSTRB[0] && (S_AXI_WDATA[1:0] != 2'b00));

    integer byte_idx;
    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            model_base_addr_reg  <= 32'd0;
            input_base_addr_reg  <= 32'd0;
            output_base_addr_reg <= 32'd0;
            layer_id_reg          <= 4'd0;
            irq_enable_reg       <= 1'b0;
        end
        else begin
            if (wr_ctrl) irq_enable_reg <= ctrl_wr_irq_enable_bit;

            if (accept_base_write) begin
                for (byte_idx = 0; byte_idx < 4; byte_idx = byte_idx + 1) begin
                    if (S_AXI_WSTRB[byte_idx]) begin
                        if (wr_model)  model_base_addr_reg[byte_idx*8 +: 8]  <= S_AXI_WDATA[byte_idx*8 +: 8];
                        if (wr_input)  input_base_addr_reg[byte_idx*8 +: 8]  <= S_AXI_WDATA[byte_idx*8 +: 8];
                        if (wr_output) output_base_addr_reg[byte_idx*8 +: 8] <= S_AXI_WDATA[byte_idx*8 +: 8];
                    end
                end
            end

            if (wr_layer_id && !busy_reg) layer_id_reg <= S_AXI_WDATA[3:0];
        end
    end

    assign o_model_base_addr  = model_base_addr_reg;
    assign o_input_base_addr  = input_base_addr_reg;
    assign o_output_base_addr = output_base_addr_reg;
    assign o_layer_id         = layer_id_reg;

    // ============================================================
    // BUSY: set immediately on an accepted START, cleared only after a
    // terminal event (DONE *or* ERROR) has been observed *and* the backend
    // has actually gone idle (i_busy==0). Neither i_done nor i_error_pulse
    // is guaranteed to land on the same cycle i_busy drops (i_busy is a
    // registered signal that drops one cycle after the combinational pulse
    // it's derived from), so a "seen terminal event, waiting for idle"
    // latch bridges that gap - generalized from GLOBAL/dummy's DONE-only
    // seen_done_pending to also cover i_error_pulse (Docs/
    // C_Phase2_Conv1_LocalBuffer_Design_v1.md S8.2: a malformed LOAD frame
    // must end in a clean STATUS.ERROR/BUSY=0, not a permanent hang).
    // ============================================================
    reg seen_terminal_pending;

    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            seen_terminal_pending <= 1'b0;
        end
        else if (o_soft_reset_pulse) begin
            seen_terminal_pending <= 1'b0;
        end
        else if (i_done || i_error_pulse) begin
            seen_terminal_pending <= 1'b1;
        end
        else if (seen_terminal_pending && !i_busy) begin
            seen_terminal_pending <= 1'b0;
        end
    end

    wire core_idle_after_terminal = (i_done || i_error_pulse || seen_terminal_pending) && !i_busy;

    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            busy_reg <= 1'b0;
        end
        else if (o_soft_reset_pulse) begin
            busy_reg <= 1'b0;
        end
        else if (o_start_pulse) begin
            busy_reg <= 1'b1;
        end
        else if (core_idle_after_terminal) begin
            busy_reg <= 1'b0;
        end
    end

    // ============================================================
    // STATUS.DONE / STATUS.ERROR (+ ERROR_CODE): sticky, write-1-to-clear,
    // but a new event always wins over a same-cycle W1C - the SET branch is
    // checked before the CLEAR branch in each if/else-if chain below.
    // ============================================================
    wire w1c_done_bit  = wr_status && S_AXI_WDATA[1];
    wire w1c_error_bit = wr_status && S_AXI_WDATA[2];

    wire new_done_event  = i_done;
    // i_error_pulse (backend LOAD-frame fault) takes priority over the two
    // CSR-local error sources, which can't coincide with it in practice
    // (those only fire on a base-addr write while the backend is busy/
    // misaligned, not from the backend's own LOAD validation).
    wire new_error_event = i_error_pulse || cfg_write_while_busy || addr_misaligned || invalid_start;
    wire [7:0] new_error_code = i_error_pulse       ? i_error_code :
                                 cfg_write_while_busy ? ERR_CFG_WRITE_WHILE_BUSY :
                                 addr_misaligned      ? ERR_ALIGNMENT :
                                 invalid_start        ? ERR_INVALID_LAYER_ID :
                                                         ERR_NONE;

    reg status_done_reg;
    reg status_error_reg;
    reg [7:0] status_error_code_reg;

    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            status_done_reg       <= 1'b0;
            status_error_reg      <= 1'b0;
            status_error_code_reg <= ERR_NONE;
        end
        else if (o_soft_reset_pulse) begin
            status_done_reg       <= 1'b0;
            status_error_reg      <= 1'b0;
            status_error_code_reg <= ERR_NONE;
        end
        else begin
            // Docs/C_CSR_Register_Map_v1.md S3: "다음 START 시에도 자동 클리어"
            // applies only to DONE, not ERROR - a fresh operation should not
            // start under a stale DONE from the previous run, but ERROR/
            // ERROR_CODE are diagnostic information that must survive until
            // explicitly W1C'd or SOFT_RESET, not be silently wiped by the
            // next START.
            if (new_done_event)
                status_done_reg <= 1'b1;
            else if (o_start_pulse || w1c_done_bit)
                status_done_reg <= 1'b0;

            if (new_error_event) begin
                status_error_reg      <= 1'b1;
                status_error_code_reg <= new_error_code;
            end
            else if (w1c_error_bit) begin
                status_error_reg      <= 1'b0;
                status_error_code_reg <= ERR_NONE;
            end
        end
    end

    // ============================================================
    // IRQ_STATUS (0x18): mirrors DONE/ERROR events, gated by IRQ_ENABLE.
    // Register-only for now - see header comment (no physical irq port).
    // Same new-event-beats-W1C priority as STATUS above.
    // ============================================================
    wire w1c_done_irq_bit  = wr_irq && S_AXI_WDATA[0];
    wire w1c_error_irq_bit = wr_irq && S_AXI_WDATA[1];

    reg irq_done_reg;
    reg irq_error_reg;

    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN == 1'b0) begin
            irq_done_reg  <= 1'b0;
            irq_error_reg <= 1'b0;
        end
        else if (o_soft_reset_pulse) begin
            irq_done_reg  <= 1'b0;
            irq_error_reg <= 1'b0;
        end
        else begin
            // Same auto-clear-on-START rule as STATUS.DONE above: DONE_IRQ
            // is residue of the same completion event, so a fresh START
            // clears it too. ERROR_IRQ is preserved like STATUS.ERROR.
            if (new_done_event && irq_enable_reg)
                irq_done_reg <= 1'b1;
            else if (o_start_pulse || w1c_done_irq_bit)
                irq_done_reg <= 1'b0;

            if (new_error_event && irq_enable_reg)
                irq_error_reg <= 1'b1;
            else if (w1c_error_irq_bit)
                irq_error_reg <= 1'b0;
        end
    end

    // ============================================================
    // Read mux
    // ============================================================
    reg [31:0] rdata_reg;
    always @(*) begin
        case (rd_sel)
            REG_CTRL:      rdata_reg = {29'd0, irq_enable_reg, 1'b0, 1'b0};
            REG_STATUS:    rdata_reg = {16'd0, status_error_code_reg, 5'd0, status_error_reg, status_done_reg, busy_reg};
            REG_MODEL:     rdata_reg = model_base_addr_reg;
            REG_INPUT:     rdata_reg = input_base_addr_reg;
            REG_OUTPUT:    rdata_reg = output_base_addr_reg;
            REG_VERSION:   rdata_reg = VERSION_ID_MAGIC;
            REG_IRQ:       rdata_reg = {30'd0, irq_error_reg, irq_done_reg};
            REG_LAYER_ID:  rdata_reg = {28'd0, layer_id_reg};
            default:       rdata_reg = 32'd0;
        endcase
    end

    assign S_AXI_RDATA = rdata_reg;

endmodule
