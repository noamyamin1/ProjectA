module rgb_frame_writer #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  enable,
    input  logic [31:0]           base_addr,

    input  logic [23:0]           s_axis_tdata,
    input  logic                  s_axis_tvalid,
    input  logic                  s_axis_tuser,
    input  logic                  s_axis_tlast,
    output logic                  s_axis_tready,

    output logic [AXI_ADDR_W-1:0] m_axi_awaddr,
    output logic [7:0]            m_axi_awlen,
    output logic [2:0]            m_axi_awsize,
    output logic [1:0]            m_axi_awburst,
    output logic                  m_axi_awvalid,
    input  logic                  m_axi_awready,

    output logic [AXI_DATA_W-1:0] m_axi_wdata,
    output logic [7:0]            m_axi_wstrb,
    output logic                  m_axi_wlast,
    output logic                  m_axi_wvalid,
    input  logic                  m_axi_wready,

    input  logic [1:0]            m_axi_bresp,
    input  logic                  m_axi_bvalid,
    output logic                  m_axi_bready
);

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_AW_REQ,
        ST_W_REQ,
        ST_B_WAIT
    } state_e;

    state_e state, next_state;

    logic [31:0] pixel_buf;
    logic        pixel_idx;
    logic [31:0] addr_offset;

    assign s_axis_tready = (state == ST_IDLE) && enable;

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
                if (enable && s_axis_tvalid && pixel_idx == 1'b1) begin
                    next_state = ST_AW_REQ;
                end
            end
            ST_AW_REQ: begin
                if (m_axi_awready) begin
                    next_state = ST_W_REQ;
                end
            end
            ST_W_REQ: begin
                if (m_axi_wready) begin
                    next_state = ST_B_WAIT;
                end
            end
            ST_B_WAIT: begin
                if (m_axi_bvalid) begin
                    next_state = ST_IDLE;
                end
            end
            default: next_state = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_idx   <= 1'b0;
            pixel_buf   <= '0;
            addr_offset <= '0;
            m_axi_wdata <= '0;
        end else begin
            if (enable && s_axis_tvalid && state == ST_IDLE) begin
                if (s_axis_tuser) begin
                    addr_offset <= '0;
                    pixel_idx   <= 1'b0;
                end

                if (pixel_idx == 1'b0) begin
                    pixel_buf <= {8'h00, s_axis_tdata};
                    pixel_idx <= 1'b1;
                end else begin
                    m_axi_wdata <= {{8'h00, s_axis_tdata}, pixel_buf};
                    pixel_idx   <= 1'b0;
                end
            end
            
            if (state == ST_B_WAIT && m_axi_bvalid) begin
                addr_offset <= addr_offset + 8;
            end
        end
    end

    assign m_axi_awaddr  = base_addr + addr_offset;
    assign m_axi_awlen   = 8'd0;
    assign m_axi_awsize  = 3'b011;
    assign m_axi_awburst = 2'b01;
    
    assign m_axi_awvalid = (state == ST_AW_REQ);
    assign m_axi_wvalid  = (state == ST_W_REQ);
    assign m_axi_wstrb   = 8'hFF;
    assign m_axi_wlast   = 1'b1;
    assign m_axi_bready  = (state == ST_B_WAIT);

endmodule