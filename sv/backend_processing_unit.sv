module backend_processing_unit #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int LABEL_W    = 9,
    parameter int IMG_W      = 1920,
    parameter int IMG_H      = 1080
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  cfg_enable,
    input  logic [31:0]           cfg_frame_base_addr,

    // Input Stream from Morphology
    input  logic                  s_axis_tdata,
    input  logic                  s_axis_tvalid,
    input  logic                  s_axis_tuser,
    input  logic                  s_axis_tlast,

    // AXI4 Master Interface (Shared DDR Access)
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
    output logic                  m_axi_bready,
    
    output logic [AXI_ADDR_W-1:0] m_axi_araddr,
    output logic [7:0]            m_axi_arlen,
    output logic [2:0]            m_axi_arsize,
    output logic [1:0]            m_axi_arburst,
    output logic                  m_axi_arvalid,
    input  logic                  m_axi_arready,
    input  logic [AXI_DATA_W-1:0] m_axi_rdata,
    input  logic                  m_axi_rlast,
    input  logic [1:0]            m_axi_rresp,
    input  logic                  m_axi_rvalid,
    output logic                  m_axi_rready,

    // Outputs to CSR
    output logic                  sts_done_flag,
    output logic [7:0]            sts_best_class_id,
    output logic [15:0]           sts_bbox_xmin,
    output logic [15:0]           sts_bbox_xmax,
    output logic [15:0]           sts_bbox_ymin,
    output logic [15:0]           sts_bbox_ymax
);

    // ==========================================
    // FSM Definitions
    // ==========================================
    typedef enum logic [2:0] {
        ST_IDLE          = 3'b000,
        ST_PASS1_STREAM  = 3'b001,
        ST_CCL_RESOLVE   = 3'b010,
        ST_PASS2_STATS   = 3'b011,
        ST_GEOMETRY      = 3'b100,
        ST_ROI_FETCH     = 3'b101,
        ST_TEMPLATE_MACH = 3'b110,
        ST_DONE          = 3'b111
    } backend_state_e;

    backend_state_e state, next_state;

    // ==========================================
    // Inter-Module Signals
    // ==========================================
    logic pass1_done, resolver_done, stats_done, geo_done, fetch_done, match_done;

    logic [LABEL_W-1:0] p2_parent_addr, p2_parent_rdata;
    
    logic [LABEL_W-1:0] geo_ram_addr;
    logic [31:0]        stats_area_rdata;
    logic [15:0]        stats_xmin_rdata, stats_xmax_rdata, stats_ymin_rdata, stats_ymax_rdata;

    logic [7:0]         fetch_gray_tdata;
    logic               fetch_gray_tvalid, fetch_gray_tlast;

    logic [9:0]         tmpl_ram_addr;
    logic [31:0]        tmpl_ram_rdata;

    logic [LABEL_W-1:0] best_label;
    logic [31:0]        best_score;

    // ==========================================
    // Internal AXI Client Signals
    // ==========================================
    // Pass 1 Writer
    logic [AXI_ADDR_W-1:0] p1_awaddr;
    logic [7:0]            p1_awlen;
    logic                  p1_awvalid;
    logic                  p1_awready;
    logic [AXI_DATA_W-1:0] p1_wdata;
    logic [7:0]            p1_wstrb;
    logic                  p1_wlast;
    logic                  p1_wvalid;
    logic                  p1_wready;
    logic                  p1_bvalid;
    logic                  p1_bready;

    // Pass 2 Reader
    logic [AXI_ADDR_W-1:0] p2_araddr;
    logic [7:0]            p2_arlen;
    logic                  p2_arvalid;
    logic                  p2_arready;
    logic [AXI_DATA_W-1:0] p2_rdata;
    logic                  p2_rlast;
    logic                  p2_rvalid;
    logic                  p2_rready;

    // ROI Fetcher Reader
    logic [AXI_ADDR_W-1:0] roi_araddr;
    logic [7:0]            roi_arlen;
    logic                  roi_arvalid;
    logic                  roi_arready;
    logic [AXI_DATA_W-1:0] roi_rdata;
    logic                  roi_rlast;
    logic                  roi_rvalid;
    logic                  roi_rready;

    // ==========================================
    // Template ROM Inference
    // ==========================================
    logic [31:0] template_rom [0:1023];
    
    initial begin
        $readmemh("templates.mem", template_rom);
    end

    always_ff @(posedge clk) begin
        tmpl_ram_rdata <= template_rom[tmpl_ram_addr];
    end

    // ==========================================
    // Main FSM
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= ST_IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE:          if (cfg_enable && s_axis_tvalid && s_axis_tuser) next_state = ST_PASS1_STREAM;
            ST_PASS1_STREAM:  if (pass1_done)    next_state = ST_CCL_RESOLVE;
            ST_CCL_RESOLVE:   if (resolver_done) next_state = ST_PASS2_STATS;
            ST_PASS2_STATS:   if (stats_done)    next_state = ST_GEOMETRY;
            ST_GEOMETRY:      if (geo_done)      next_state = (best_label == 0) ? ST_DONE : ST_ROI_FETCH;
            ST_ROI_FETCH:     if (fetch_done)    next_state = ST_TEMPLATE_MACH;
            ST_TEMPLATE_MACH: if (match_done)    next_state = ST_DONE;
            ST_DONE:          if (!cfg_enable)   next_state = ST_IDLE;
        endcase
    end

    // ==========================================
    // AXI Arbiter / Multiplexer
    // ==========================================
    
    // Constant AR/AW attributes
    assign m_axi_awsize  = 3'b011; // 8 bytes (64-bit)
    assign m_axi_awburst = 2'b01;  // INCR
    assign m_axi_arsize  = 3'b011;
    assign m_axi_arburst = 2'b01;

    always_comb begin
        // Default assignments (Isolation)
        m_axi_awaddr  = '0; m_axi_awlen   = '0; m_axi_awvalid = 1'b0;
        m_axi_wdata   = '0; m_axi_wstrb   = '0; m_axi_wlast   = 1'b0; m_axi_wvalid  = 1'b0;
        m_axi_bready  = 1'b0;
        
        m_axi_araddr  = '0; m_axi_arlen   = '0; m_axi_arvalid = 1'b0;
        m_axi_rready  = 1'b0;

        p1_awready = 1'b0; p1_wready = 1'b0; p1_bvalid = 1'b0;
        p2_arready = 1'b0; p2_rdata  = '0;   p2_rlast  = 1'b0; p2_rvalid = 1'b0;
        roi_arready= 1'b0; roi_rdata = '0;   roi_rlast = 1'b0; roi_rvalid= 1'b0;

        case (state)
            ST_PASS1_STREAM: begin
                // Write Channel -> Pass 1 Writer
                m_axi_awaddr  = p1_awaddr;
                m_axi_awlen   = p1_awlen;
                m_axi_awvalid = p1_awvalid;
                p1_awready    = m_axi_awready;
                
                m_axi_wdata   = p1_wdata;
                m_axi_wstrb   = p1_wstrb;
                m_axi_wlast   = p1_wlast;
                m_axi_wvalid  = p1_wvalid;
                p1_wready     = m_axi_wready;
                
                m_axi_bready  = p1_bready;
                p1_bvalid     = m_axi_bvalid;
            end
            
            ST_PASS2_STATS: begin
                // Read Channel -> Pass 2 Reader
                m_axi_araddr  = p2_araddr;
                m_axi_arlen   = p2_arlen;
                m_axi_arvalid = p2_arvalid;
                p2_arready    = m_axi_arready;
                
                m_axi_rready  = p2_rready;
                p2_rdata      = m_axi_rdata;
                p2_rlast      = m_axi_rlast;
                p2_rvalid     = m_axi_rvalid;
            end
            
            ST_ROI_FETCH: begin
                // Read Channel -> ROI Fetcher
                m_axi_araddr  = roi_araddr;
                m_axi_arlen   = roi_arlen;
                m_axi_arvalid = roi_arvalid;
                roi_arready   = m_axi_arready;
                
                m_axi_rready  = roi_rready;
                roi_rdata     = m_axi_rdata;
                roi_rlast     = m_axi_rlast;
                roi_rvalid    = m_axi_rvalid;
            end
            
            default: ;
        endcase
    end

    // ==========================================
    // Instances
    // ==========================================
    
    ccl_engine #(
        .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H), .LABEL_W(LABEL_W)
    ) u_ccl_engine (
        .clk             (clk),
        .rst_n           (rst_n),
        .s_axis_tdata    (s_axis_tdata),
        .s_axis_tvalid   (s_axis_tvalid),
        .s_axis_tuser    (s_axis_tuser),
        .s_axis_tlast    (s_axis_tlast),
        .pass1_done      (pass1_done),
        .resolver_done   (resolver_done),
        .p2_parent_addr  (p2_parent_addr),
        .p2_parent_rdata (p2_parent_rdata)
    );

    ccl_stats_collector #(
        .LABEL_W(LABEL_W)
    ) u_ccl_stats (
        .clk             (clk),
        .rst_n           (rst_n),
        
        // Driven by Pass 2 Read logic (Simplified mapped to p2_rdata)
        .s_axis_label    (p2_rdata[LABEL_W-1:0]), 
        .s_axis_tvalid   (p2_rvalid), 
        .s_axis_tuser    (p2_rvalid && p2_araddr == cfg_frame_base_addr), // Mock SOF
        .s_axis_tlast    (p2_rlast),
        
        .parent_addr     (p2_parent_addr),
        .parent_rdata    (p2_parent_rdata),
        .geo_ram_addr    (geo_ram_addr),
        .out_area        (stats_area_rdata),
        .out_xmin        (stats_xmin_rdata),
        .out_xmax        (stats_xmax_rdata),
        .out_ymin        (stats_ymin_rdata),
        .out_ymax        (stats_ymax_rdata),
        .stats_done      (stats_done)
    );

    geometry_filter #(
        .LABEL_W(LABEL_W)
    ) u_geometry_filter (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (state == ST_GEOMETRY),
        .max_label   (9'd511),
        .min_area_th (32'd300),
        .ram_addr    (geo_ram_addr),
        .area_rdata  (stats_area_rdata),
        .xmin_rdata  (stats_xmin_rdata),
        .xmax_rdata  (stats_xmax_rdata),
        .ymin_rdata  (stats_ymin_rdata),
        .ymax_rdata  (stats_ymax_rdata),
        .done        (geo_done),
        .best_label  (best_label),
        .best_xmin   (sts_bbox_xmin),
        .best_xmax   (sts_bbox_xmax),
        .best_ymin   (sts_bbox_ymin),
        .best_ymax   (sts_bbox_ymax)
    );

    roi_fetcher_axi_master #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .FRAME_W(IMG_W)
    ) u_roi_fetcher (
        .clk                (clk),
        .rst_n              (rst_n),
        .start              (state == ST_ROI_FETCH),
        .base_addr          (cfg_frame_base_addr),
        .roi_xmin           (sts_bbox_xmin),
        .roi_xmax           (sts_bbox_xmax),
        .roi_ymin           (sts_bbox_ymin),
        .roi_ymax           (sts_bbox_ymax),
        
        .m_axi_araddr       (roi_araddr),
        .m_axi_arlen        (roi_arlen),
        .m_axi_arsize       (),
        .m_axi_arburst      (),
        .m_axi_arvalid      (roi_arvalid),
        .m_axi_arready      (roi_arready),
        
        .m_axi_rdata        (roi_rdata),
        .m_axi_rresp        (2'b00),
        .m_axi_rlast        (roi_rlast),
        .m_axi_rvalid       (roi_rvalid),
        .m_axi_rready       (roi_rready),
        
        .m_axis_gray_tdata  (fetch_gray_tdata),
        .m_axis_gray_tvalid (fetch_gray_tvalid),
        .m_axis_gray_tlast  (fetch_gray_tlast),
        .fetch_done         (fetch_done)
    );

    template_matching_engine u_matcher (
        .clk                (clk),
        .rst_n              (rst_n),
        .s_axis_gray_tdata  (fetch_gray_tdata),
        .s_axis_gray_tvalid (fetch_gray_tvalid),
        .s_axis_gray_tlast  (fetch_gray_tlast),
        .template_ram_addr  (tmpl_ram_addr),
        .template_ram_rdata (tmpl_ram_rdata),
        .match_done         (match_done),
        .best_class_id      (sts_best_class_id),
        .best_score         (best_score)
    );

    assign sts_done_flag = (state == ST_DONE);

endmodule