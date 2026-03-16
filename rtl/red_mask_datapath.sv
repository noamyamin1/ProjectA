`timescale 1ns / 1ps

module red_mask_datapath #(
    parameter int TDATA_W = 24
)(
    input  logic               clk,
    input  logic               rst_n,
    
    // Configuration from CSR
    input  logic [7:0]         min_red_val,
    input  logic [2:0]         margin_shift,
    
    // AXI4-Stream Slave Interface (Input RGB)
    input  logic [TDATA_W-1:0] s_axis_tdata,
    input  logic               s_axis_tvalid,
    output logic               s_axis_tready,
    input  logic               s_axis_tuser,
    input  logic               s_axis_tlast,
    
    // AXI4-Stream Master Interface (Output 1-bit Mask)
    output logic               m_axis_tdata,
    output logic               m_axis_tvalid,
    output logic               m_axis_tuser,
    output logic               m_axis_tlast
);

    // Assuming RGB888 format
    localparam int R_MSB = 23;
    localparam int R_LSB = 16;
    localparam int G_MSB = 15;
    localparam int G_LSB = 8;
    localparam int B_MSB = 7;
    localparam int B_LSB = 0;

    // We can always accept data in this pipeline 
    // (Assuming downstream Morphology block uses Line Buffers and doesn't stall)
    assign s_axis_tready = 1'b1;

    // ==========================================
    // Pipeline Stage 1: Margin & Additions
    // ==========================================
    logic [7:0] stg1_r, stg1_g, stg1_b;
    logic [8:0] stg1_g_plus_margin;
    logic [8:0] stg1_b_plus_margin;
    
    logic       stg1_tvalid;
    logic       stg1_tuser;
    logic       stg1_tlast;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg1_r             <= 8'd0;
            stg1_g             <= 8'd0;
            stg1_b             <= 8'd0;
            stg1_g_plus_margin <= 9'd0;
            stg1_b_plus_margin <= 9'd0;
            stg1_tvalid        <= 1'b0;
            stg1_tuser         <= 1'b0;
            stg1_tlast         <= 1'b0;
        end else begin
            stg1_tvalid <= s_axis_tvalid;
            
            if (s_axis_tvalid) begin
                automatic logic [7:0] r_in = s_axis_tdata[R_MSB:R_LSB];
                automatic logic [7:0] g_in = s_axis_tdata[G_MSB:G_LSB];
                automatic logic [7:0] b_in = s_axis_tdata[B_MSB:B_LSB];
                automatic logic [7:0] margin = r_in >> margin_shift;

                stg1_r <= r_in;
                stg1_g <= g_in;
                stg1_b <= b_in;
                
                // 9-bit addition to prevent overflow
                stg1_g_plus_margin <= {1'b0, g_in} + {1'b0, margin};
                stg1_b_plus_margin <= {1'b0, b_in} + {1'b0, margin};
                
                stg1_tuser <= s_axis_tuser;
                stg1_tlast <= s_axis_tlast;
            end
        end
    end

    // ==========================================
    // Pipeline Stage 2: Comparisons & Mask Gen
    // ==========================================
    logic stg2_mask;
    logic stg2_tvalid;
    logic stg2_tuser;
    logic stg2_tlast;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg2_mask   <= 1'b0;
            stg2_tvalid <= 1'b0;
            stg2_tuser  <= 1'b0;
            stg2_tlast  <= 1'b0;
        end else begin
            stg2_tvalid <= stg1_tvalid;
            
            if (stg1_tvalid) begin
                automatic logic cond1_dom_r_vs_g;
                automatic logic cond2_dom_r_vs_b;
                automatic logic cond3_orange_killer;
                automatic logic cond4_min_red;
                
                cond1_dom_r_vs_g    = ({1'b0, stg1_r} > stg1_g_plus_margin);
                cond2_dom_r_vs_b    = ({1'b0, stg1_r} > stg1_b_plus_margin);
                cond3_orange_killer = ({1'b0, stg1_g} < stg1_b_plus_margin);
                cond4_min_red       = (stg1_r > min_red_val);
                
                stg2_mask  <= cond1_dom_r_vs_g & cond2_dom_r_vs_b & cond3_orange_killer & cond4_min_red;
                stg2_tuser <= stg1_tuser;
                stg2_tlast <= stg1_tlast;
            end
        end
    end

    // ==========================================
    // Output Assignments
    // ==========================================
    assign m_axis_tdata  = stg2_mask;
    assign m_axis_tvalid = stg2_tvalid;
    assign m_axis_tuser  = stg2_tuser;
    assign m_axis_tlast  = stg2_tlast;

endmodule