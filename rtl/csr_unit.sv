`timescale 1ns / 1ps

module csr_unit #(
    parameter int ADDR_W = 12,
    parameter int DATA_W = 32
)(
    input  logic              clk,
    input  logic              rst_n,

    // AXI4-Lite Interface
    input  logic [ADDR_W-1:0] s_axi_awaddr,
    input  logic              s_axi_awvalid,
    output logic              s_axi_awready,
    input  logic [DATA_W-1:0] s_axi_wdata,
    input  logic [3:0]        s_axi_wstrb,
    input  logic              s_axi_wvalid,
    output logic              s_axi_wready,
    output logic [1:0]        s_axi_bresp,
    output logic              s_axi_bvalid,
    input  logic              s_axi_bready,
    input  logic [ADDR_W-1:0] s_axi_araddr,
    input  logic              s_axi_arvalid,
    output logic              s_axi_arready,
    output logic [DATA_W-1:0] s_axi_rdata,
    output logic [1:0]        s_axi_rresp,
    output logic              s_axi_rvalid,
    input  logic              s_axi_rready,

    // Hardware Interface (Config)
    output logic [7:0]        cfg_min_red_val,
    output logic [2:0]        cfg_margin_shift,
    output logic [31:0]       cfg_frame_base_addr,
    output logic              cfg_enable,

    // Hardware Interface (Status)
    input  logic              sts_done_flag,
    input  logic [7:0]        sts_best_class_id,
    input  logic [15:0]       sts_bbox_xmin,
    input  logic [15:0]       sts_bbox_xmax,
    input  logic [15:0]       sts_bbox_ymin,
    input  logic [15:0]       sts_bbox_ymax,

    // Interrupt
    output logic              irq
);

    // Register definitions
    logic        reg_enable;
    logic [7:0]  reg_min_red_val;
    logic [2:0]  reg_margin_shift;
    logic [31:0] reg_frame_base_addr;
    
    // Interrupt logic
    logic irq_ff;
    logic irq_ack;

    // AXI4-Lite signals
    logic aw_en;
    logic axi_awready;
    logic axi_wready;
    logic axi_bvalid;
    logic axi_arready;
    logic axi_rvalid;
    logic [DATA_W-1:0] axi_rdata;

    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = 2'b00; // OKAY
    assign s_axi_bvalid  = axi_bvalid;
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata   = axi_rdata;
    assign s_axi_rresp   = 2'b00; // OKAY
    assign s_axi_rvalid  = axi_rvalid;

    // Output assignments
    assign cfg_enable          = reg_enable;
    assign cfg_min_red_val     = reg_min_red_val;
    assign cfg_margin_shift    = reg_margin_shift;
    assign cfg_frame_base_addr = reg_frame_base_addr;
    assign irq                 = irq_ff;

    // Write Address & Data Handshake
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            aw_en       <= 1'b1;
        end else begin
            if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                axi_awready <= 1'b1;
                aw_en       <= 1'b0;
            end else if (s_axi_bready && axi_bvalid) begin
                axi_awready <= 1'b0;
                aw_en       <= 1'b1;
            end else begin
                axi_awready <= 1'b0;
            end

            if (~axi_wready && s_axi_wvalid && s_axi_awvalid && aw_en) begin
                axi_wready <= 1'b1;
            end else begin
                axi_wready <= 1'b0;
            end
        end
    end

    // Write Response Handshake
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_bvalid <= 1'b0;
        end else begin
            if (axi_awready && s_axi_awvalid && ~axi_bvalid && axi_wready && s_axi_wvalid) begin
                axi_bvalid <= 1'b1;
            end else if (s_axi_bready && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    // Write Logic & Interrupt Ack
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_enable          <= 1'b0;
            reg_min_red_val     <= 8'd15;
            reg_margin_shift    <= 3'd3;
            reg_frame_base_addr <= 32'd0;
            irq_ack             <= 1'b0;
        end else begin
            irq_ack <= 1'b0; 
            if (axi_wready && s_axi_wvalid && axi_awready && s_axi_awvalid) begin
                case (s_axi_awaddr[7:2])
                    6'h00: begin 
                        if (s_axi_wstrb[0]) reg_enable <= s_axi_wdata[0];
                        if (s_axi_wstrb[0] && s_axi_wdata[1]) irq_ack <= 1'b1; 
                    end
                    6'h04: begin
                        if (s_axi_wstrb[0]) reg_min_red_val  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) reg_margin_shift <= s_axi_wdata[10:8];
                    end
                    6'h05: begin
                        if (s_axi_wstrb[0]) reg_frame_base_addr[7:0]   <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) reg_frame_base_addr[15:8]  <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) reg_frame_base_addr[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) reg_frame_base_addr[31:24] <= s_axi_wdata[31:24];
                    end
                    default: ;
                endcase
            end
        end
    end

    // Read Address Handshake
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
        end else begin
            if (~axi_arready && s_axi_arvalid) begin
                axi_arready <= 1'b1;
            end else begin
                axi_arready <= 1'b0;
            end

            if (axi_arready && s_axi_arvalid && ~axi_rvalid) begin
                axi_rvalid <= 1'b1;
            end else if (axi_rvalid && s_axi_rready) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    // Read Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_rdata <= '0;
        end else begin
            if (axi_arready && s_axi_arvalid && ~axi_rvalid) begin
                case (s_axi_araddr[7:2])
                    6'h00: axi_rdata <= {31'd0, reg_enable};
                    6'h01: axi_rdata <= {16'd0, sts_best_class_id, 7'd0, sts_done_flag};
                    6'h02: axi_rdata <= {sts_bbox_xmax, sts_bbox_xmin};
                    6'h03: axi_rdata <= {sts_bbox_ymax, sts_bbox_ymin};
                    6'h04: axi_rdata <= {21'd0, reg_margin_shift, reg_min_red_val};
                    6'h05: axi_rdata <= reg_frame_base_addr;
                    default: axi_rdata <= '0;
                endcase
            end
        end
    end

    // Interrupt Generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            irq_ff <= 1'b0;
        end else begin
            if (sts_done_flag) begin
                irq_ff <= 1'b1;
            end else if (irq_ack) begin
                irq_ff <= 1'b0;
            end
        end
    end

endmodule