`timescale 1ns / 1ps

module ccl_pass1_labeler #(
    parameter int IMG_WIDTH = 1920,
    parameter int LABEL_W   = 9
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic               s_axis_tdata,
    input  logic               s_axis_tvalid,
    input  logic               s_axis_tuser,
    input  logic               s_axis_tlast,

    output logic [LABEL_W-1:0] m_axis_tdata,
    output logic               m_axis_tvalid,
    output logic               m_axis_tuser,
    output logic               m_axis_tlast,

    output logic               parent_we,
    output logic [LABEL_W-1:0] parent_addr,
    output logic [LABEL_W-1:0] parent_wdata
);

    // ==========================================
    // Block 1: Internal Signals & Line Buffer
    // ==========================================
    logic [LABEL_W-1:0] line_buf [0:IMG_WIDTH-1];
    logic [11:0]        x_cnt;
    
    logic               mask_q;
    logic               valid_q;
    logic               user_q;
    logic               last_q;
    
    logic [LABEL_W-1:0] up_label;
    logic [LABEL_W-1:0] left_label;
    logic [LABEL_W-1:0] next_label_cnt;
    logic [LABEL_W-1:0] out_label;

    // ==========================================
    // Block 2: X Coordinate Counter
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_cnt <= '0;
        end else if (s_axis_tvalid) begin
            if (s_axis_tlast)
                x_cnt <= '0;
            else
                x_cnt <= x_cnt + 1;
        end
    end

    // ==========================================
    // Block 3: Line Buffer Read & Input Delay
    // ==========================================
    always_ff @(posedge clk) begin
        if (s_axis_tvalid) begin
            up_label <= line_buf[x_cnt];
            mask_q   <= s_axis_tdata;
            valid_q  <= 1'b1;
            user_q   <= s_axis_tuser;
            last_q   <= s_axis_tlast;
        end else begin
            valid_q  <= 1'b0;
        end
    end

    // ==========================================
    // Block 4: Label Resolution Logic
    // ==========================================
    always_comb begin
        out_label    = '0;
        parent_we    = 1'b0;
        parent_addr  = '0;
        parent_wdata = '0;

        if (valid_q && mask_q) begin
            if (left_label == '0 && up_label == '0) begin
                out_label = next_label_cnt;
                
                parent_we    = 1'b1;
                parent_addr  = next_label_cnt;
                parent_wdata = next_label_cnt;
                
            end else if (left_label != '0 && up_label == '0) begin
                out_label = left_label;
                
            end else if (left_label == '0 && up_label != '0) begin
                out_label = up_label;
                
            end else begin
                if (left_label < up_label) begin
                    out_label    = left_label;
                    parent_we    = 1'b1;
                    parent_addr  = up_label;
                    parent_wdata = left_label;
                end else if (left_label > up_label) begin
                    out_label    = up_label;
                    parent_we    = 1'b1;
                    parent_addr  = left_label;
                    parent_wdata = up_label;
                end else begin
                    out_label = left_label;
                end
            end
        end
    end

    // ==========================================
    // Block 5: State Updates (Counters & Buffers)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            left_label     <= '0;
            next_label_cnt <= 9'd1;
        end else begin
            if (valid_q) begin
                line_buf[x_cnt > 0 ? x_cnt - 1 : IMG_WIDTH - 1] <= out_label;
                
                if (last_q) begin
                    left_label <= '0;
                end else begin
                    left_label <= out_label;
                end

                if (mask_q && left_label == '0 && up_label == '0 && next_label_cnt < {LABEL_W{1'b1}}) begin
                    next_label_cnt <= next_label_cnt + 1;
                end
            end

            if (valid_q && user_q) begin
                next_label_cnt <= 9'd1;
                left_label     <= '0;
            end
        end
    end

    // ==========================================
    // Block 6: Output Stream Assignments
    // ==========================================
    assign m_axis_tdata  = out_label;
    assign m_axis_tvalid = valid_q;
    assign m_axis_tuser  = user_q;
    assign m_axis_tlast  = last_q;

endmodule