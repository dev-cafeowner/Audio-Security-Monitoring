`timescale 1ns / 1ps
`include "npu_defs.vh"

// Unified NPU integration top.  It contains exactly one b_compute_top_16;
// all layer-specific state remains in selected frontend FSMs.
module neural_processing_unit_unified_6_0 #(
    parameter integer C_S00_AXI_DATA_WIDTH  = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH  = 5,
    parameter integer C_S00_AXIS_TDATA_WIDTH = 32,
    parameter integer C_M00_AXIS_TDATA_WIDTH = 32
) (
    input  wire s00_axi_aclk, input wire s00_axi_aresetn,
    input wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_awaddr,
    input wire [2:0] s00_axi_awprot, input wire s00_axi_awvalid, output wire s00_axi_awready,
    input wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_wdata,
    input wire [(C_S00_AXI_DATA_WIDTH/8)-1:0] s00_axi_wstrb,
    input wire s00_axi_wvalid, output wire s00_axi_wready,
    output wire [1:0] s00_axi_bresp, output wire s00_axi_bvalid, input wire s00_axi_bready,
    input wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_araddr,
    input wire [2:0] s00_axi_arprot, input wire s00_axi_arvalid, output wire s00_axi_arready,
    output wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_rdata, output wire [1:0] s00_axi_rresp,
    output wire s00_axi_rvalid, input wire s00_axi_rready,

    output wire s00_axis_tready, input wire [C_S00_AXIS_TDATA_WIDTH-1:0] s00_axis_tdata,
    input wire [(C_S00_AXIS_TDATA_WIDTH/8)-1:0] s00_axis_tkeep,
    input wire s00_axis_tlast, input wire s00_axis_tvalid,
    output wire m00_axis_tvalid, output wire [C_M00_AXIS_TDATA_WIDTH-1:0] m00_axis_tdata,
    output wire [(C_M00_AXIS_TDATA_WIDTH/8)-1:0] m00_axis_tkeep,
    output wire m00_axis_tlast, input wire m00_axis_tready
);
    localparam [1:0] OP_CONV=2'd0, OP_FC=2'd1, OP_GLOBAL=2'd2;
    wire clk=s00_axi_aclk, rst_n=s00_axi_aresetn;

    // CSR: only the AXIS stream is data-plane; model/base pointers remain
    // scheduler-owned metadata in this first unified top.
    wire csr_start, csr_reset;
    wire [3:0] csr_layer_id;
    wire [31:0] unused_model_base, unused_input_base, unused_output_base;
    wire top_busy, top_done, top_error;
    wire [7:0] top_error_code;
    npu_unified_csr_axi_lite #(.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH), .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)) u_csr (
        .o_start_pulse(csr_start), .o_soft_reset_pulse(csr_reset), .i_busy(top_busy),
        .i_done(top_done), .i_error_pulse(top_error), .i_error_code(top_error_code),
        .o_layer_id(csr_layer_id), .o_model_base_addr(unused_model_base),
        .o_input_base_addr(unused_input_base), .o_output_base_addr(unused_output_base),
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(rst_n), .S_AXI_AWADDR(s00_axi_awaddr),
        .S_AXI_AWPROT(s00_axi_awprot), .S_AXI_AWVALID(s00_axi_awvalid), .S_AXI_AWREADY(s00_axi_awready),
        .S_AXI_WDATA(s00_axi_wdata), .S_AXI_WSTRB(s00_axi_wstrb), .S_AXI_WVALID(s00_axi_wvalid),
        .S_AXI_WREADY(s00_axi_wready), .S_AXI_BRESP(s00_axi_bresp), .S_AXI_BVALID(s00_axi_bvalid),
        .S_AXI_BREADY(s00_axi_bready), .S_AXI_ARADDR(s00_axi_araddr), .S_AXI_ARPROT(s00_axi_arprot),
        .S_AXI_ARVALID(s00_axi_arvalid), .S_AXI_ARREADY(s00_axi_arready), .S_AXI_RDATA(s00_axi_rdata),
        .S_AXI_RRESP(s00_axi_rresp), .S_AXI_RVALID(s00_axi_rvalid), .S_AXI_RREADY(s00_axi_rready)
    );

    // Descriptor is read from the CSR-selected ID for admission and again
    // from active_layer_id after START.  Busy-write protection makes that
    // active descriptor immutable for the complete operation.
    wire csr_desc_valid; wire [1:0] csr_op_mode; wire csr_pool;
    wire [`NPU_CFG_C_W-1:0] csr_cin, csr_cout; wire [4:0] csr_groups;
    wire [`NPU_CFG_T_W-1:0] csr_tconv, csr_tout; wire [31:0] csr_last;
    wire [`NPU_CFG_DIM_W-1:0] csr_fc_in, csr_fc_out;
    npu_unified_descriptor_rom u_csr_desc (.i_layer_id(csr_layer_id), .o_valid(csr_desc_valid), .o_op_mode(csr_op_mode),
        .o_pool_enable(csr_pool), .o_cin(csr_cin), .o_cout(csr_cout), .o_num_oc_group(csr_groups),
        .o_conv_out_len(csr_tconv), .o_post_out_len(csr_tout), .o_last_output_group(csr_last),
        .o_conv_activation_beats(), .o_fc_in_dim(csr_fc_in), .o_fc_out_dim(csr_fc_out),
        .o_fc_chunk_count(), .o_fc_last_valid_taps());

    wire conv1_done, convn_done, global_done, fc1_done;
    wire conv1_err, convn_err, fc1_err, conv_param_err;
    wire [7:0] conv1_err_code, convn_err_code, fc1_err_code, conv_param_err_code;
    wire active_valid; wire [3:0] active_layer_id; wire frontend_start;
    wire start_accept, invalid_start;
    wire [1:0] active_kind = (active_layer_id == 4'd9) ? OP_GLOBAL :
                             ((active_layer_id == 4'd10 || active_layer_id == 4'd11) ? OP_FC : OP_CONV);
    wire sel_conv1 = active_valid && (active_layer_id == 4'd0);
    wire sel_convn = active_valid && (active_layer_id >= 4'd1) && (active_layer_id <= 4'd8);
    wire sel_global = active_valid && (active_layer_id == 4'd9);
    wire sel_fc1 = active_valid && (active_layer_id == 4'd10);
    wire sel_fc2 = active_valid && (active_layer_id == 4'd11);
    wire sel_fc = sel_fc1 || sel_fc2;
    wire selected_done = sel_conv1 ? conv1_done : sel_convn ? convn_done :
                         sel_global ? global_done : sel_fc ? fc1_done : 1'b0;
    wire selected_error = (sel_conv1 || sel_convn) && conv_param_err ? 1'b1 :
                          sel_conv1 ? conv1_err : sel_convn ? convn_err :
                           sel_fc ? fc1_err : 1'b0;
    wire [7:0] selected_error_code = (sel_conv1 || sel_convn) && conv_param_err ? conv_param_err_code :
                                     sel_conv1 ? conv1_err_code : sel_convn ? convn_err_code :
                                      sel_fc ? fc1_err_code : 8'h00;

    // Keep immediate frontend-error abort semantics at u_layer_ctrl, but
    // isolate the CSR's wide status capture from frontend counter/compare
    // logic by registering the reporting event for one cycle.
    npu_unified_error_reporter u_error_reporter (
        .clk(clk), .rst_n(rst_n), .i_flush(csr_reset),
        .i_error_pulse(selected_error), .i_error_code(selected_error_code),
        .o_error_pulse(top_error), .o_error_code(top_error_code)
    );
    npu_unified_layer_controller u_layer_ctrl (
        .clk(clk), .rst_n(rst_n), .i_layer_id(csr_layer_id), .i_layer_id_valid(csr_desc_valid),
        .i_start_pulse(csr_start), .i_soft_reset_pulse(csr_reset), .i_terminal_done(selected_done),
        .i_terminal_error(selected_error), .o_active_valid(active_valid), .o_active_layer_id(active_layer_id),
        .o_frontend_start_pulse(frontend_start), .o_start_accept(start_accept), .o_invalid_start(invalid_start));

    wire active_desc_valid; wire [1:0] active_op_mode; wire active_pool;
    wire [`NPU_CFG_C_W-1:0] active_cin, active_cout; wire [4:0] active_groups;
    wire [`NPU_CFG_T_W-1:0] active_tconv, active_tout; wire [31:0] active_last;
    wire [`NPU_CFG_DIM_W-1:0] active_fc_in, active_fc_out;
    wire [31:0] active_conv_activation_beats;
    wire [`NPU_CFG_DIM_W-1:0] active_fc_chunk_count;
    wire [1:0] active_fc_last_valid_taps;
    npu_unified_descriptor_rom u_active_desc (.i_layer_id(active_layer_id), .o_valid(active_desc_valid), .o_op_mode(active_op_mode),
        .o_pool_enable(active_pool), .o_cin(active_cin), .o_cout(active_cout), .o_num_oc_group(active_groups),
        .o_conv_out_len(active_tconv), .o_post_out_len(active_tout), .o_last_output_group(active_last),
        .o_conv_activation_beats(active_conv_activation_beats),
        .o_fc_in_dim(active_fc_in), .o_fc_out_dim(active_fc_out),
        .o_fc_chunk_count(active_fc_chunk_count), .o_fc_last_valid_taps(active_fc_last_valid_taps));

    // AXIS wires for the four mutually-exclusive frontends.
    wire c1_sready,c1_mvalid,c1_mlast, cn_sready,cn_mvalid,cn_mlast, gl_sready,gl_mvalid,gl_mlast, f1_sready,f1_mvalid,f1_mlast;
    wire [31:0] c1_mdata,cn_mdata,gl_mdata,f1_mdata;
    wire [3:0] c1_mkeep,cn_mkeep,gl_mkeep,f1_mkeep;

    // Common B-core boundary, one signal set per frontend.
    wire c1_rst,c1_start,c1_pool,c1_av,c1_ar,c1_wv,c1_wr,c1_or,c1_ov,c1_busy,c1_bdone;
    wire cn_rst,cn_start,cn_pool,cn_av,cn_ar,cn_wv,cn_wr,cn_or,cn_ov,cn_busy,cn_bdone;
    wire gl_start,gl_gv,gl_gr,gl_or,gl_ov,gl_busy,gl_bdone;
    wire f1_rst,f1_start,f1_fc1,f1_av,f1_ar,f1_wv,f1_wr,f1_or,f1_ov;
    wire signed [15:0] c1_x0,c1_x1,c1_x2,f1_x0,f1_x1,f1_x2;
    wire [`NPU_PIN4_X_BUS_W-1:0] cn_x;
    wire [`NPU_WEIGHT_BUS_W-1:0] c1_w,f1_w;
    wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] cn_w;
    wire [`NPU_BIAS_BUS_W-1:0] c1_bias,cn_bias,f1_bias;
    wire [`NPU_RSHIFT_BUS_W-1:0] c1_rsh,cn_rsh,f1_rsh;
    wire c1_rshift_valid, cn_rshift_valid, f1_rshift_valid;
    wire [`NPU_GLOBAL_BUS_W-1:0] gl_data;
    wire [`NPU_CFG_C_W-1:0] c1_cin,c1_cout,cn_cin,cn_cout;
    wire [`NPU_CFG_T_W-1:0] c1_tconv,c1_tout,cn_tconv,cn_tout;
    wire [`NPU_CFG_DIM_W-1:0] f1_in,f1_out;
    wire [`NPU_GROUP_W-1:0] b_result_group,b_out_group;
    wire [`NPU_OUT_BUS_W-1:0] b_out_data;
    wire [15:0] b_out_mask;
    wire b_act_ready,b_weight_ready,b_global_ready,b_out_valid,b_busy,b_done;
    wire legacy_src_ready, legacy_b_valid;
    wire [3*`NPU_DATA_W-1:0] legacy_x;
    wire [`NPU_WEIGHT_BUS_W-1:0] legacy_weight;
    wire [`NPU_BIAS_BUS_W-1:0] legacy_bias;
    wire result_b_ready, result_valid;
    wire [2:0] result_owner;
    wire [`NPU_OUT_BUS_W-1:0] result_data;
    wire [15:0] result_mask;
    wire [`NPU_GROUP_W-1:0] result_group;
    wire [`NPU_CFG_C_W-1:0] b_conv_cin_idx,b_conv_oc_idx;
    wire [`NPU_CFG_T_W-1:0] b_conv_time_idx;
    wire [`NPU_CFG_C_W-1:0] cn_param_cin_idx;
    wire [`NPU_GROUP_W-1:0] cn_param_group_idx;

    // One physical loader/store serves every convolution layer. Conv1 is the
    // Cin=1/group=4 descriptor; Conv2..9 use the active ROM descriptor.
    wire conv_param_loading, conv_params_ready, conv_param_sready;
    wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] conv_param_weight;
    wire [`NPU_WEIGHT_BUS_W-1:0] conv_param_weight_cin0;
    wire [`NPU_BIAS_BUS_W-1:0] conv_param_bias;
    wire [`NPU_BIAS_BUS_W-1:0] conv_param_bias_prefetch;
    wire [`NPU_RSHIFT_BUS_W-1:0] conv_param_rshift;
    wire conv_param_weight_valid, conv_param_bias_valid, conv_param_rshift_valid;
    wire conv_selected = sel_conv1 || sel_convn;
    wire [`NPU_CFG_C_W-1:0] conv_param_cin = sel_conv1 ? {{(`NPU_CFG_C_W-1){1'b0}},1'b1} : active_cin;
    wire [4:0] conv_param_groups = sel_conv1 ? 5'd4 : active_groups;
    wire [`NPU_CFG_C_W-1:0] conv_param_issue_cin = sel_convn ? cn_param_cin_idx : b_conv_cin_idx;
    wire [`NPU_GROUP_W-1:0] conv_param_issue_group = sel_convn ? cn_param_group_idx : b_conv_oc_idx[`NPU_GROUP_W-1:0];

    npu_unified_conv_param_system #(.MAX_CIN(128),.MAX_OC_GROUP(16)) u_conv_params (
        .clk(clk),.rst_n(rst_n),.i_start(frontend_start&&conv_selected),
        .i_flush(csr_reset),.i_cin(conv_param_cin),
        .i_num_oc_group(conv_param_groups),.o_loading(conv_param_loading),
        .o_params_ready_pulse(conv_params_ready),.o_error_pulse(conv_param_err),
        .o_error_code(conv_param_err_code),.s_axis_tready(conv_param_sready),
        .s_axis_tdata(s00_axis_tdata),.s_axis_tkeep(s00_axis_tkeep),
        .s_axis_tlast(s00_axis_tlast),.s_axis_tvalid(s00_axis_tvalid),
        .i_issue_cin_idx(conv_param_issue_cin),.i_issue_group_idx(conv_param_issue_group),
        .i_result_group_idx(b_result_group),.o_weight(conv_param_weight),
        .o_bias(conv_param_bias),.o_bias_prefetch(conv_param_bias_prefetch),.o_rshift(conv_param_rshift),
        .o_weight_valid(conv_param_weight_valid),.o_bias_valid(conv_param_bias_valid),
        .o_rshift_valid(conv_param_rshift_valid)
    );

    // Conv1 reuses the Pin4 parameter store but consumes only Cin slice 0.
    genvar cp_lane;
    generate
        for (cp_lane=0; cp_lane<`NPU_LANES; cp_lane=cp_lane+1) begin : GEN_CONV1_WEIGHT_VIEW
            assign conv_param_weight_cin0[`NPU_WEIGHT_BIT_OFS(cp_lane,0) +: 16] =
                conv_param_weight[`NPU_PIN4_WEIGHT_BIT_OFS(cp_lane,0,0) +: 16];
            assign conv_param_weight_cin0[`NPU_WEIGHT_BIT_OFS(cp_lane,1) +: 16] =
                conv_param_weight[`NPU_PIN4_WEIGHT_BIT_OFS(cp_lane,0,1) +: 16];
            assign conv_param_weight_cin0[`NPU_WEIGHT_BIT_OFS(cp_lane,2) +: 16] =
                conv_param_weight[`NPU_PIN4_WEIGHT_BIT_OFS(cp_lane,0,2) +: 16];
        end
    endgenerate

    npu_conv1_frontend #(.USE_SHARED_PARAMS(1)) u_conv1 (.clk(clk),.rst_n(rst_n),.i_start_pulse(frontend_start&&sel_conv1),.i_soft_reset_pulse(csr_reset),.o_busy(c1_busy),.o_done(conv1_done),.o_error_pulse(conv1_err),.o_error_code(conv1_err_code),
        .s_axis_tready(c1_sready),.s_axis_tdata(s00_axis_tdata),.s_axis_tkeep(s00_axis_tkeep),.s_axis_tlast(s00_axis_tlast),.s_axis_tvalid(s00_axis_tvalid&&!conv_param_loading),.m_axis_tvalid(c1_mvalid),.m_axis_tdata(c1_mdata),.m_axis_tkeep(c1_mkeep),.m_axis_tlast(c1_mlast),.m_axis_tready(m00_axis_tready),
        .o_b_reset_request(c1_rst),.o_b_start(c1_start),.o_b_pool_enable(c1_pool),.o_b_conv_cin(c1_cin),.o_b_conv_cout(c1_cout),.o_b_conv_out_len(c1_tconv),.o_b_conv_post_out_len(c1_tout),.o_b_act_valid(c1_av),.i_b_act_ready(c1_ar),.o_b_x0(c1_x0),.o_b_x1(c1_x1),.o_b_x2(c1_x2),.o_b_weight_valid(c1_wv),.i_b_weight_ready(c1_wr),.o_b_weight(c1_w),.o_b_bias(c1_bias),.o_b_rshift(c1_rsh),.o_b_rshift_valid(c1_rshift_valid),.i_b_result_group_idx(b_result_group),.i_b_out_valid(c1_ov),.o_b_out_ready(c1_or),.i_b_out_data(result_data),.i_b_busy(b_busy),.i_b_done(c1_bdone),.i_b_conv_oc_group_idx(b_conv_oc_idx),
        .i_shared_params_ready(conv_params_ready),.i_shared_weight(conv_param_weight_cin0),.i_shared_bias(conv_param_bias),.i_shared_rshift(conv_param_rshift),.i_shared_weight_valid(conv_param_weight_valid),.i_shared_bias_valid(conv_param_bias_valid),.i_shared_rshift_valid(conv_param_rshift_valid));

    npu_convn_frontend #(.MAX_CIN(128),.MAX_OC_GROUP(16),.USE_SHARED_PARAMS(1)) u_convn (.clk(clk),.rst_n(rst_n),.i_start_pulse(frontend_start&&sel_convn),.i_soft_reset_pulse(csr_reset),.o_busy(cn_busy),.o_done(convn_done),.o_error_pulse(convn_err),.o_error_code(convn_err_code),
        .i_cin(active_cin),.i_num_oc_group(active_groups),.i_conv_out_len(active_tconv),.i_pool_enable(active_pool),.i_conv_post_out_len(active_tout),.i_last_output_group(active_last),.i_activation_beats(active_conv_activation_beats),.s_axis_tready(cn_sready),.s_axis_tdata(s00_axis_tdata),.s_axis_tkeep(s00_axis_tkeep),.s_axis_tlast(s00_axis_tlast),.s_axis_tvalid(s00_axis_tvalid&&!conv_param_loading),.m_axis_tvalid(cn_mvalid),.m_axis_tdata(cn_mdata),.m_axis_tkeep(cn_mkeep),.m_axis_tlast(cn_mlast),.m_axis_tready(m00_axis_tready),
        .o_b_reset_request(cn_rst),.o_b_start(cn_start),.o_b_pool_enable(cn_pool),.o_b_conv_cin(cn_cin),.o_b_conv_cout(cn_cout),.o_b_conv_out_len(cn_tconv),.o_b_conv_post_out_len(cn_tout),.o_b_act_valid(cn_av),.i_b_act_ready(cn_ar),.o_b_x(cn_x),.o_b_weight_valid(cn_wv),.i_b_weight_ready(cn_wr),.o_b_weight(cn_w),.o_b_bias(cn_bias),.o_b_rshift(cn_rsh),.o_b_rshift_valid(cn_rshift_valid),.i_b_result_group_idx(b_result_group),.i_b_out_valid(cn_ov),.o_b_out_ready(cn_or),.i_b_out_data(result_data),.i_b_busy(b_busy),.i_b_done(cn_bdone),.i_b_conv_cin_idx(b_conv_cin_idx),.i_b_conv_oc_group_idx(b_conv_oc_idx),.i_b_conv_time_idx(b_conv_time_idx),
        .i_shared_params_ready(conv_params_ready),.i_shared_weight(conv_param_weight),.i_shared_bias(conv_param_bias_prefetch),.i_shared_rshift(conv_param_rshift),.i_shared_weight_valid(conv_param_weight_valid),.i_shared_bias_valid(conv_param_bias_valid),.i_shared_rshift_valid(conv_param_rshift_valid),.o_param_issue_cin_idx(cn_param_cin_idx),.o_param_issue_group_idx(cn_param_group_idx),.dbg_step_fire(),.dbg_time_idx(),.dbg_oc_group_idx(),.dbg_cin_idx(),.dbg_x0(),.dbg_x1(),.dbg_x2(),.dbg_sel_weight());

    npu_global_frontend u_global (.clk(clk),.rst_n(rst_n),.i_start_pulse(frontend_start&&sel_global),.i_soft_reset_pulse(csr_reset),.o_busy(gl_busy),.o_done(global_done),.s_axis_tready(gl_sready),.s_axis_tdata(s00_axis_tdata),.s_axis_tkeep(s00_axis_tkeep),.s_axis_tlast(s00_axis_tlast),.s_axis_tvalid(s00_axis_tvalid),.m_axis_tvalid(gl_mvalid),.m_axis_tdata(gl_mdata),.m_axis_tkeep(gl_mkeep),.m_axis_tlast(gl_mlast),.m_axis_tready(m00_axis_tready),.o_b_start(gl_start),.o_b_global_valid(gl_gv),.o_b_global_data(gl_data),.i_b_global_ready(gl_gr),.i_b_out_valid(gl_ov),.o_b_out_ready(gl_or),.i_b_out_data(result_data),.i_b_out_lane_valid_mask(result_mask),.i_b_out_group_idx(result_group),.i_b_busy(b_busy),.i_b_done(gl_bdone));

    // One maximum-size FC frontend serves FC1 and FC2.  Its descriptor
    // controls frame lengths and B dimensions; only one FC tile store exists.
    npu_fc_frontend #(.FC_IN(512),.FC_OUT(527),.FC1_MODE(0),.USE_RUNTIME_DESC(1)) u_fc1 (.clk(clk),.rst_n(rst_n),.i_start_pulse(frontend_start&&sel_fc),.i_soft_reset_pulse(csr_reset),.i_desc_fc1_mode(sel_fc1),.i_desc_fc_in_dim(active_fc_in),.i_desc_fc_out_dim(active_fc_out),.o_busy(),.o_done(fc1_done),.o_error_pulse(fc1_err),.o_error_code(fc1_err_code),.s_axis_tready(f1_sready),.s_axis_tdata(s00_axis_tdata),.s_axis_tkeep(s00_axis_tkeep),.s_axis_tlast(s00_axis_tlast),.s_axis_tvalid(s00_axis_tvalid),.m_axis_tvalid(f1_mvalid),.m_axis_tdata(f1_mdata),.m_axis_tkeep(f1_mkeep),.m_axis_tlast(f1_mlast),.m_axis_tready(m00_axis_tready),.o_b_reset_request(f1_rst),.o_b_start(f1_start),.o_b_fc1_mode(f1_fc1),.o_b_fc_in_dim(f1_in),.o_b_fc_out_dim(f1_out),.o_b_act_valid(f1_av),.i_b_act_ready(f1_ar),.o_b_x0(f1_x0),.o_b_x1(f1_x1),.o_b_x2(f1_x2),.o_b_weight_valid(f1_wv),.i_b_weight_ready(f1_wr),.o_b_weight(f1_w),.o_b_bias(f1_bias),.o_b_rshift(f1_rsh),.o_b_rshift_valid(f1_rshift_valid),.i_b_result_group_idx(b_result_group),.i_b_out_valid(f1_ov),.o_b_out_ready(f1_or),.i_b_out_data(result_data),.i_b_out_lane_mask(result_mask),.i_b_out_group_idx(result_group));

    // Selected AXIS owner only.  Inactive frontend streams cannot handshake.
    assign s00_axis_tready = conv_selected && conv_param_loading ? conv_param_sready :
                             sel_conv1 ? c1_sready : sel_convn ? cn_sready : sel_global ? gl_sready : sel_fc ? f1_sready : 1'b0;
    assign m00_axis_tvalid = sel_conv1 ? c1_mvalid : sel_convn ? cn_mvalid : sel_global ? gl_mvalid : sel_fc ? f1_mvalid : 1'b0;
    assign m00_axis_tdata = sel_conv1 ? c1_mdata : sel_convn ? cn_mdata : sel_global ? gl_mdata : f1_mdata;
    assign m00_axis_tkeep = sel_conv1 ? c1_mkeep : sel_convn ? cn_mkeep : sel_global ? gl_mkeep : f1_mkeep;
    assign m00_axis_tlast = sel_conv1 ? c1_mlast : sel_convn ? cn_mlast : sel_global ? gl_mlast : f1_mlast;

    reg b_rst_n;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) b_rst_n <= 1'b0;
        // Operation-boundary cleanup is mandatory even after a nominally
        // successful frontend completion.  The unified result tracker can
        // finish after the externally required result beats have drained
        // while shared_mac_acc_core's private controller still reports busy.
        // Without this one-cycle reset, the next frontend START is accepted
        // by the outer core but rejected by shared_mac_acc_core (!o_busy),
        // permanently deasserting B ready and stalling the activation DMA.
        // selected_done is the final frontend output handshake, so no result
        // or operand belonging to the completed operation remains observable.
        else if (csr_reset || selected_done ||
                 (sel_conv1&&c1_rst) || (sel_convn&&cn_rst) ||
                 (sel_fc&&f1_rst)) b_rst_n <= 1'b0;
        else b_rst_n <= 1'b1;
    end
    // Capture the output owner once at a START admission.  Do not derive B's
    // output-ready path from active_layer_id: that creates a long registered
    // layer-ID -> B ready chain -> frontend storage-CE path in the unified
    // top.  The owner remains stable until the next accepted operation/reset.
    localparam [2:0] OUT_OWNER_NONE   = 3'd0;
    localparam [2:0] OUT_OWNER_CONV1  = 3'd1;
    localparam [2:0] OUT_OWNER_CONVN  = 3'd2;
    localparam [2:0] OUT_OWNER_GLOBAL = 3'd3;
    localparam [2:0] OUT_OWNER_FC     = 3'd4;
    reg [2:0] b_out_owner;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || csr_reset)
            b_out_owner <= OUT_OWNER_NONE;
        else if (start_accept) begin
            if (csr_layer_id == 4'd0)
                b_out_owner <= OUT_OWNER_CONV1;
            else if (csr_layer_id >= 4'd1 && csr_layer_id <= 4'd8)
                b_out_owner <= OUT_OWNER_CONVN;
            else if (csr_layer_id == 4'd9)
                b_out_owner <= OUT_OWNER_GLOBAL;
            else
                b_out_owner <= OUT_OWNER_FC;
        end
    end
    wire result_owner_ready = (result_owner == OUT_OWNER_CONV1)  ? c1_or :
                              (result_owner == OUT_OWNER_CONVN)  ? cn_or :
                              (result_owner == OUT_OWNER_GLOBAL) ? gl_or :
                              (result_owner == OUT_OWNER_FC)     ? f1_or : 1'b0;
    // All B responses are owner-routed.  Even a ready response is not safe to
    // broadcast here: FC tile-store control can use it as a CE condition while
    // an unrelated Conv frontend is draining output.  The owner is registered
    // at START, so this demux never depends on active_layer_id in a datapath.
    wire own_conv1  = (b_out_owner == OUT_OWNER_CONV1);
    wire own_convn  = (b_out_owner == OUT_OWNER_CONVN);
    wire own_global = (b_out_owner == OUT_OWNER_GLOBAL);
    wire own_fc     = (b_out_owner == OUT_OWNER_FC);
    // Conv1/FC retain a narrow elastic boundary so B's ready cone cannot reach
    // their storage CEs. ConvN already has a two-entry, lane-local Pin4 FIFO;
    // connect that FIFO directly and avoid a second 3072-bit flat register
    // bank, which was the dominant full-SoC placement/routing hotspot.
    assign c1_ar=legacy_src_ready && own_conv1; assign c1_wr=legacy_src_ready && own_conv1; assign c1_ov=result_valid && (result_owner == OUT_OWNER_CONV1);  assign c1_bdone=b_done && own_conv1;
    assign cn_ar=b_act_ready && own_convn; assign cn_wr=b_weight_ready && own_convn; assign cn_ov=result_valid && (result_owner == OUT_OWNER_CONVN);  assign cn_bdone=b_done && own_convn;
    assign gl_gr=b_global_ready && own_global; assign gl_ov=result_valid && (result_owner == OUT_OWNER_GLOBAL); assign gl_bdone=b_done && own_global;
    assign f1_ar=legacy_src_ready && own_fc; assign f1_wr=legacy_src_ready && own_fc; assign f1_ov=result_valid && (result_owner == OUT_OWNER_FC);

    wire legacy_src_valid = sel_conv1 ? (c1_av && c1_wv) :
                            sel_fc    ? (f1_av && f1_wv) : 1'b0;
    // Frontend error requests already pass through b_rst_n above before they
    // reset the shared B core.  Use that registered reset state to flush the
    // wide operand buffer as well.  Driving i_flush directly from cn_rst made
    // ConvN's accept-time 32-bit length/error compare feed every operand
    // payload CE in one cycle (run_expected_beats_r -> bias_reg/CE).  B is
    // held in reset while !b_rst_n, so clearing this one-entry buffer on the
    // following clock preserves abort semantics without exposing that wide
    // combinational cone.  CSR soft reset remains immediate here.
    wire operand_flush = csr_reset || !b_rst_n;

    npu_unified_operand_buffer u_operand_buffer (
        .clk(clk), .rst_n(rst_n), .i_flush(operand_flush),
        .i_src_valid(legacy_src_valid),
        .o_src_ready(legacy_src_ready),
        .i_src_x(sel_conv1 ? {c1_x2,c1_x1,c1_x0} : {f1_x2,f1_x1,f1_x0}),
        .i_src_weight(sel_conv1 ? c1_w : f1_w),
        .i_src_bias(sel_conv1 ? c1_bias : f1_bias),
        .o_b_valid(legacy_b_valid),
        .i_b_act_ready(b_act_ready), .i_b_weight_ready(b_weight_ready),
        .o_b_x(legacy_x),
        .o_b_weight(legacy_weight), .o_b_bias(legacy_bias)
    );

    wire [`NPU_PIN4_X_BUS_W-1:0] legacy_x_pin4 =
        {{(`NPU_PIN4_X_BUS_W-3*`NPU_DATA_W){1'b0}},legacy_x};
    wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] legacy_weight_pin4;
    genvar operand_lane;
    generate
        for (operand_lane=0; operand_lane<`NPU_LANES; operand_lane=operand_lane+1) begin : GEN_LEGACY_WEIGHT_PIN0
            assign legacy_weight_pin4[`NPU_PIN4_WEIGHT_BIT_OFS(operand_lane,0,0) +: 16] = legacy_weight[`NPU_WEIGHT_BIT_OFS(operand_lane,0) +: 16];
            assign legacy_weight_pin4[`NPU_PIN4_WEIGHT_BIT_OFS(operand_lane,0,1) +: 16] = legacy_weight[`NPU_WEIGHT_BIT_OFS(operand_lane,1) +: 16];
            assign legacy_weight_pin4[`NPU_PIN4_WEIGHT_BIT_OFS(operand_lane,0,2) +: 16] = legacy_weight[`NPU_WEIGHT_BIT_OFS(operand_lane,2) +: 16];
            assign legacy_weight_pin4[(operand_lane*`NPU_PIN*`NPU_TAPS+`NPU_TAPS)*`NPU_WEIGHT_W +: 9*`NPU_WEIGHT_W] = 144'd0;
        end
    endgenerate

    wire b_operand_valid = own_convn ? (cn_av && cn_wv) : legacy_b_valid;
    wire [`NPU_PIN4_X_BUS_W-1:0] b_operand_x = own_convn ? cn_x : legacy_x_pin4;
    wire [`NPU_PIN4_WEIGHT_BUS_W-1:0] b_operand_weight = own_convn ? cn_w : legacy_weight_pin4;
    wire [`NPU_BIAS_BUS_W-1:0] b_operand_bias = own_convn ? cn_bias : legacy_bias;

    // Rshift is result-aligned by the frozen B/C contract and therefore must
    // not be captured with issue-side operands.  ConvN may briefly deassert
    // valid while its group-boundary BRAM prefetch completes; the other
    // frontends provide their rshift combinationally and are always valid.
    wire [`NPU_RSHIFT_BUS_W-1:0] result_aligned_rshift =
        sel_conv1 ? c1_rsh : sel_convn ? cn_rsh : sel_fc ? f1_rsh :
        {`NPU_RSHIFT_BUS_W{1'b0}};
    wire result_aligned_rshift_valid = sel_conv1 ? c1_rshift_valid : sel_convn ? cn_rshift_valid :
                                              sel_fc ? f1_rshift_valid : 1'b1;

    // Exactly one B-core.  Non-selected operand buses are irrelevant because
    // their corresponding valid is forced low.
    b_compute_top_16 #(.RSHIFT_VALID_GATING(1)) u_bcore (.clk(clk),.rst_n(b_rst_n),
        .i_op_mode(active_kind==OP_CONV?`NPU_OP_CONV:active_kind==OP_FC?`NPU_OP_FC:`NPU_OP_GLOBAL),
        .i_start(sel_conv1?c1_start:sel_convn?cn_start:sel_global?gl_start:sel_fc?f1_start:1'b0),
        .i_pool_enable(sel_conv1?c1_pool:sel_convn?cn_pool:1'b0),.i_fc1_mode(sel_fc?f1_fc1:1'b0),
        .i_conv_cin(sel_conv1?c1_cin:cn_cin),.i_conv_cout(sel_conv1?c1_cout:cn_cout),.i_conv_out_len(sel_conv1?c1_tconv:cn_tconv),.i_conv_post_out_len(sel_conv1?c1_tout:cn_tout),
        .i_fc_in_dim(f1_in),.i_fc_out_dim(f1_out),.i_fc_chunk_count(active_fc_chunk_count),.i_fc_last_valid_taps(active_fc_last_valid_taps),
        .i_act_valid(b_operand_valid),.o_act_ready(b_act_ready),
        .i_x(b_operand_x),
        .i_weight_valid(b_operand_valid),.o_weight_ready(b_weight_ready),
        .i_weight(b_operand_weight),.i_bias(b_operand_bias),
        .i_rshift(result_aligned_rshift),.i_rshift_valid(result_aligned_rshift_valid),
        .o_result_group_idx(b_result_group),.o_result_time_idx(),.i_global_valid(sel_global?gl_gv:1'b0),.o_global_ready(b_global_ready),.i_global_data(gl_data),.o_global_group_idx(),.o_global_time_idx(),.o_output_op_mode(),.o_valid(b_out_valid),.i_out_ready(result_b_ready),.o_data(b_out_data),.o_lane_valid_mask(b_out_mask),.o_group_idx(b_out_group),.o_busy(b_busy),.o_done(b_done),.o_conv_cin_idx(b_conv_cin_idx),.o_conv_oc_group_idx(b_conv_oc_idx),.o_conv_time_idx(b_conv_time_idx),.o_fc_chunk_idx(),.o_fc_out_group_idx());

    npu_unified_result_buffer u_result_buffer (
        .clk(clk), .rst_n(rst_n), .i_flush(csr_reset), .i_capture_owner(b_out_owner),
        .i_b_valid(b_out_valid), .o_b_ready(result_b_ready), .i_b_data(b_out_data),
        .i_b_lane_mask(b_out_mask), .i_b_group_idx(b_out_group), .o_valid(result_valid),
        .o_owner(result_owner), .o_data(result_data), .o_lane_mask(result_mask),
        .o_group_idx(result_group), .i_owner_ready(result_owner_ready)
    );

    assign top_done = selected_done;
    assign top_busy = active_valid || b_busy;
endmodule
