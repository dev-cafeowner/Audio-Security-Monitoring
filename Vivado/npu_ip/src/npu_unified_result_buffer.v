`include "npu_defs.vh"

// One-entry result boundary between the shared B core and the selected
// frontend serializer. B sees only registered empty credit; frontend
// backpressure is confined to dequeue.
module npu_unified_result_buffer (
    input wire clk, input wire rst_n, input wire i_flush,
    input wire [2:0] i_capture_owner,
    input wire i_b_valid, output wire o_b_ready,
    input wire [`NPU_OUT_BUS_W-1:0] i_b_data,
    input wire [15:0] i_b_lane_mask,
    input wire [`NPU_GROUP_W-1:0] i_b_group_idx,
    output wire o_valid, output wire [2:0] o_owner,
    output wire [`NPU_OUT_BUS_W-1:0] o_data,
    output wire [15:0] o_lane_mask,
    output wire [`NPU_GROUP_W-1:0] o_group_idx,
    input wire i_owner_ready
);
    reg valid_reg;
    reg [2:0] owner_reg;
    reg [`NPU_OUT_BUS_W-1:0] data_reg;
    reg [15:0] lane_mask_reg;
    reg [`NPU_GROUP_W-1:0] group_idx_reg;

    // No fall-through replacement: B-side ready never depends on frontend
    // combinational ready.
    assign o_b_ready = !valid_reg;
    assign o_valid = valid_reg;
    assign o_owner = owner_reg;
    assign o_data = data_reg;
    assign o_lane_mask = lane_mask_reg;
    assign o_group_idx = group_idx_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || i_flush) begin
            valid_reg <= 1'b0;
            owner_reg <= 3'd0;
        end else if (!valid_reg && i_b_valid) begin
            valid_reg <= 1'b1;
            owner_reg <= i_capture_owner;
            data_reg <= i_b_data;
            lane_mask_reg <= i_b_lane_mask;
            group_idx_reg <= i_b_group_idx;
        end else if (valid_reg && i_owner_ready) begin
            valid_reg <= 1'b0;
        end
    end
endmodule
