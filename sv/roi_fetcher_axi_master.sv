module roi_fetcher_axi_master #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 32,
    parameter int FRAME_W    = 1920
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  start,
    input  logic [31:0]           base_addr,
    input  logic [15:0]           roi_xmin,
    input  logic [15:0]           roi_xmax,
    input  logic [15:0]           roi_ymin,
    input  logic [15:0]           roi_ymax,

    output logic [AXI_ADDR_W-1:0] m_axi_araddr,
    output logic [7:0]            m_axi_arlen,
    output logic [2:0]            m_axi_arsize,
    output logic [1:0]            m_axi_arburst,
    output logic                  m_axi_arvalid,
    input  logic                  m_axi_arready,

    input  logic [AXI_DATA_W-1:0] m_axi_rdata,
    input  logic [1:0]            m_axi_rresp,
    input  logic                  m_axi_rlast,
    input  logic                  m_axi_rvalid,
    output logic                  m_axi_rready,

    output logic [7:0]            m_axis_gray_tdata,
    output logic                  m_axis_gray_tvalid,
    output logic                  m_axis_gray_tlast,

    output logic                  fetch_done
);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_CALC,
        ST_AR_REQ,
        ST_R_WAIT,
        ST_NEXT,
        ST_DONE
    } state_e;

    state_e state, next_state;

    logic [6:0] x_dst_cnt;
    logic [6:0] y_dst_cnt;
    logic [15:0] roi_w;
    logic [15:0] roi_h;
    
    logic [31:0] x_src;
    logic [31:0] y_src;
    logic [AXI_ADDR_W-1:0] target_addr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start)
                    next_state = ST_CALC;
            end
            ST_CALC: begin
                next_state = ST_AR_REQ;
            end
            ST_AR_REQ: begin
                if (m_axi_arready)
                    next_state = ST_R_WAIT;
            end
            ST_R_WAIT: begin
                if (m_axi_rvalid)
                    next_state = ST_NEXT;
            end
            ST_NEXT: begin
                if (y_dst_cnt == 7'd64)
                    next_state = ST_DONE;
                else
                    next_state = ST_CALC;
            end
            ST_DONE: begin
                if (!start)
                    next_state = ST_IDLE;
            end
            default: next_state = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_dst_cnt <= '0;
            y_dst_cnt <= '0;
            roi_w     <= '0;
            roi_h     <= '0;
        end else begin
            if (state == ST_IDLE && start) begin
                x_dst_cnt <= '0;
                y_dst_cnt <= '0;
                roi_w     <= roi_xmax - roi_xmin + 16'd1;
                roi_h     <= roi_ymax - roi_ymin + 16'd1;
            end else if (state == ST_NEXT) begin
                if (x_dst_cnt == 7'd63) begin
                    x_dst_cnt <= '0;
                    y_dst_cnt <= y_dst_cnt + 1;
                end else begin
                    x_dst_cnt <= x_dst_cnt + 1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_src <= '0;
            y_src <= '0;
            target_addr <= '0;
        end else if (state == ST_CALC) begin
            x_src <= roi_xmin + ((x_dst_cnt * roi_w) >> 6);
            y_src <= roi_ymin + ((y_dst_cnt * roi_h) >> 6);
            
            target_addr <= base_addr + (((roi_ymin + ((y_dst_cnt * roi_h) >> 6)) * FRAME_W) + (roi_xmin + ((x_dst_cnt * roi_w) >> 6))) * 4;
        end
    end

    assign m_axi_araddr  = target_addr;
    assign m_axi_arlen   = 8'd0; 
    assign m_axi_arsize  = 3'b010; 
    assign m_axi_arburst = 2'b01; 
    assign m_axi_arvalid = (state == ST_AR_REQ);
    assign m_axi_rready  = (state == ST_R_WAIT);

    logic [7:0] r_chan, g_chan, b_chan;
    logic [15:0] gray_val;

    assign r_chan = m_axi_rdata[23:16];
    assign g_chan = m_axi_rdata[15:8];
    assign b_chan = m_axi_rdata[7:0];
    
    assign gray_val = (8'd77 * r_chan + 8'd150 * g_chan + 8'd29 * b_chan) >> 8;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axis_gray_tdata  <= '0;
            m_axis_gray_tvalid <= 1'b0;
            m_axis_gray_tlast  <= 1'b0;
        end else begin
            m_axis_gray_tvalid <= (state == ST_R_WAIT && m_axi_rvalid);
            if (state == ST_R_WAIT && m_axi_rvalid) begin
                m_axis_gray_tdata <= gray_val[7:0];
                m_axis_gray_tlast <= (x_dst_cnt == 7'd63 && y_dst_cnt == 7'd63);
            end
        end
    end

    assign fetch_done = (state == ST_DONE);

endmodule