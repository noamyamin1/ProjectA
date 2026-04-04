`timescale 1ns / 1ps

module backend_processing_unit #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 64,
    parameter int LABEL_W    = 16,
    parameter int IMG_W      = 2872,
    parameter int IMG_H      = 1617
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  cfg_enable,
    input  logic [31:0]           cfg_frame_base_addr,
    input  logic                  frame_written,

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
    output logic [15:0]           sts_bbox_ymax,
    output logic [7:0]            detected_id,
    output logic                  bbox_valid
);
    // Detected ID and bbox_valid logic for road_sign_detector
    logic [7:0] detected_id_reg, next_detected_id;
    logic       bbox_valid_reg, next_bbox_valid;
    assign detected_id = detected_id_reg;
    assign bbox_valid  = bbox_valid_reg;


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
    logic pass1_done, resolver_done, stats_done, stats_init_done, geo_done, fetch_done, match_done;
    logic [LABEL_W-1:0] p2_parent_addr, p2_parent_rdata;
    logic [LABEL_W-1:0] stats_parent_addr;
    logic [LABEL_W-1:0] geo_ram_addr;
    logic [31:0]        stats_area_rdata, stats_perim_rdata;
    logic [15:0]        stats_xmin_rdata, stats_xmax_rdata, stats_ymin_rdata, stats_ymax_rdata;
    logic [7:0]         fetch_gray_tdata;
    logic               fetch_gray_tvalid, fetch_gray_tlast;
    logic [10:0]        tmpl_ram_addr;
    logic [31:0]        tmpl_ram_rdata;
    logic [LABEL_W-1:0] best_label;
    logic [31:0]        best_score;
    logic               obj_valid;

    // Output from CCL Pass 1
    logic [LABEL_W-1:0] p1_axis_tdata;
    logic               p1_axis_tvalid;
    logic               p1_axis_tlast;

    // Internal AXI Signals
    logic [AXI_ADDR_W-1:0] p1_awaddr;
    logic [7:0]            p1_awlen;
    logic                  p1_awvalid, p1_awready;
    logic [AXI_DATA_W-1:0] p1_wdata;
    logic [7:0]            p1_wstrb;
    logic                  p1_wlast, p1_wvalid, p1_wready, p1_bready;

    logic [AXI_ADDR_W-1:0] p2_araddr;
    logic [7:0]            p2_arlen;
    logic                  p2_arvalid, p2_arready;
    logic [AXI_DATA_W-1:0] p2_rdata;
    logic                  p2_rlast, p2_rvalid, p2_rready;

    logic [AXI_ADDR_W-1:0] roi_araddr;
    logic [7:0]            roi_arlen;
    logic                  roi_arvalid, roi_arready;
    logic [AXI_DATA_W-1:0] roi_rdata;
    logic                  roi_rlast, roi_rvalid, roi_rready;

    logic [7:0] matcher_best_class_id; // Wire from Matcher
    logic [7:0] latched_best_class_id; // Register for CSR
    localparam int NO_SIGN_SCORE_THRESHOLD = 230;
    logic [15:0] stream_row_cnt;
    logic frame_eof_seen;
    logic geo_done_seen;
    logic geo_obj_seen;
    logic stats_clear;

    // ==========================================
    // DMA WRITER: Pack Pass 1 Labels to DDR
    // ==========================================
    logic [31:0] p1_addr_cnt;
    logic [1:0]  p1_pack_cnt;
    logic [63:0] p1_pack_data;

    assign p1_awlen  = 8'd0;
    assign p1_bready = 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p1_awvalid  <= 1'b0;
            p1_wvalid   <= 1'b0;
            p1_addr_cnt <= '0;
            p1_pack_cnt <= '0;
            p1_pack_data<= '0;
        end else if (state == ST_IDLE) begin
            p1_addr_cnt <= cfg_frame_base_addr + 32'h0200_0000;
            p1_pack_cnt <= '0;
            p1_awvalid  <= 1'b0;
            p1_wvalid   <= 1'b0;
        end else if (state == ST_PASS1_STREAM) begin
            if (p1_awvalid && p1_awready) p1_awvalid <= 1'b0;
            if (p1_wvalid && p1_wready)   p1_wvalid  <= 1'b0;

            if (p1_axis_tvalid) begin
                p1_pack_data[p1_pack_cnt * 16 +: 16] <= p1_axis_tdata;
                
                if (p1_pack_cnt == 2'd3 || p1_axis_tlast) begin
                    p1_awvalid  <= 1'b1;
                    p1_awaddr   <= p1_addr_cnt;
                    p1_wvalid   <= 1'b1;
                    
                    case (p1_pack_cnt)
                        2'd0: p1_wdata <= {48'd0, p1_axis_tdata};
                        2'd1: p1_wdata <= {32'd0, p1_axis_tdata, p1_pack_data[15:0]};
                        2'd2: p1_wdata <= {16'd0, p1_axis_tdata, p1_pack_data[31:0]};
                        2'd3: p1_wdata <= {p1_axis_tdata, p1_pack_data[47:0]};
                    endcase
                    
                    p1_wstrb    <= 8'hFF;
                    p1_wlast    <= 1'b1;
                    p1_addr_cnt <= p1_addr_cnt + 8;
                    p1_pack_cnt <= '0;
                end else begin
                    p1_pack_cnt <= p1_pack_cnt + 1;
                end
            end
        end
    end

    // ==========================================
    // DMA READER: Unpack Labels from DDR to Pass 2
    // ==========================================
    typedef enum logic [1:0] { R_IDLE, R_REQ, R_WAIT, R_UNPACK } p2_read_state_e;
    p2_read_state_e p2_rstate;

    logic [31:0] p2_addr_cnt;
    logic [31:0] p2_pixel_cnt;
    logic [1:0]  p2_unpack_cnt;
    logic [63:0] p2_unpack_data;
    
    logic [15:0] p2_stream_tdata;
    logic        p2_stream_tvalid;
    logic        p2_stream_tlast;
    logic [15:0] p2_label_raw;
    logic [15:0] p2_label_raw_d1;
    logic        p2_label_valid;
    logic        p2_label_last;
    logic        p2_label_valid_d1;
    logic        p2_label_last_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p2_rstate        <= R_IDLE;
            p2_addr_cnt      <= '0;
            p2_pixel_cnt     <= '0;
            p2_arvalid       <= 1'b0;
            p2_rready        <= 1'b0;
        end else if (state == ST_IDLE) begin
            p2_rstate        <= R_IDLE;
            p2_addr_cnt      <= cfg_frame_base_addr + 32'h0200_0000;
            p2_pixel_cnt     <= '0;
            p2_arvalid       <= 1'b0;
            p2_rready        <= 1'b0;
        end else if (state == ST_PASS2_STATS) begin
            case (p2_rstate)
                R_IDLE: begin
                    p2_label_valid   <= 1'b0;
                    if (!stats_init_done) begin
                        p2_rstate <= R_IDLE;
                    end else if (p2_pixel_cnt < (IMG_W * IMG_H) && !p2_arvalid) begin
                        p2_rstate  <= R_REQ;
                        p2_arvalid <= 1'b1;
                        p2_araddr  <= p2_addr_cnt;
                        p2_arlen   <= 8'd0;
                        p2_addr_cnt<= p2_addr_cnt + 8;
                    end
                end
                R_REQ: begin
                    if (p2_arvalid && p2_arready) begin
                        p2_arvalid <= 1'b0;
                        p2_rready  <= 1'b1;
                        p2_rstate  <= R_WAIT;
                    end
                end
                R_WAIT: begin
                    if (p2_rvalid && p2_rready) begin
                        p2_unpack_data <= p2_rdata;
                        p2_unpack_cnt  <= 2'd0;
                        p2_rready      <= 1'b0;
                        p2_rstate      <= R_UNPACK;
                    end
                end
                R_UNPACK: begin
                    p2_label_valid <= 1'b1;
                    p2_label_raw   <= p2_unpack_data[p2_unpack_cnt * 16 +: 16];
                    p2_label_last  <= (((p2_pixel_cnt % IMG_W) == (IMG_W - 1)) ? 1'b1 : 1'b0);
                    p2_pixel_cnt     <= p2_pixel_cnt + 1;

                    if (p2_unpack_cnt == 2'd3 || ((p2_pixel_cnt % IMG_W) == (IMG_W - 1))) begin
                        p2_unpack_cnt <= '0;
                        p2_rstate     <= R_IDLE;
                    end else begin
                        p2_unpack_cnt <= p2_unpack_cnt + 1;
                    end
                end
            endcase
        end else begin
            p2_label_valid   <= 1'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p2_label_valid_d1 <= 1'b0;
            p2_label_last_d1  <= 1'b0;
            p2_label_raw_d1   <= 16'd0;
        end else begin
            p2_label_valid_d1 <= p2_label_valid;
            p2_label_last_d1  <= p2_label_last;
            if (p2_label_valid) begin
                p2_label_raw_d1 <= p2_label_raw;
            end
        end
    end

    assign p2_parent_addr  = p2_label_raw;
    assign p2_stream_tdata = (p2_label_raw_d1 == 16'd0) ? 16'd0 : p2_parent_rdata;
    assign p2_stream_tvalid = p2_label_valid_d1;
    assign p2_stream_tlast  = p2_label_last_d1;

    // ==========================================
    // HARDWARE PROBES & PRINTS (BULLETPROOF)
    // ==========================================
    // All internal prints removed per project policy.

    // ==========================================
    // Template ROM Inference
    // ==========================================
    logic [31:0] template_rom [0:1023];
    initial begin
        $readmemh("data/templates.mem", template_rom);
    end
    always_comb begin
        tmpl_ram_rdata = template_rom[tmpl_ram_addr];
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
            
            // Wait naturally for stats to complete
            ST_PASS2_STATS:   if (stats_done || p2_pixel_cnt >= (IMG_W * IMG_H)) next_state = ST_GEOMETRY;
            
            ST_GEOMETRY: begin
                if (geo_done && !geo_done_seen) begin
                    next_state = ST_GEOMETRY;
                end else if (geo_done) begin
                    if ((obj_valid || geo_obj_seen) && (best_label != 0)) begin
                        next_state = frame_written ? ST_ROI_FETCH : ST_GEOMETRY;
                    end else begin
                        next_state = ST_DONE;
                    end
                end
            end
            ST_ROI_FETCH:     if (fetch_done)    next_state = ST_TEMPLATE_MACH;
            ST_TEMPLATE_MACH: if (match_done)    next_state = ST_DONE;
            ST_DONE: begin
                if (!cfg_enable || frame_eof_seen) next_state = ST_IDLE;
            end
            default: next_state = ST_IDLE;
        endcase
    end

    assign stats_clear = (state == ST_IDLE) && cfg_enable && s_axis_tvalid && s_axis_tuser;

    // ==========================================
    // AXI Arbiter / Multiplexer
    // ==========================================
    assign m_axi_awsize  = 3'b011; // 8 bytes (64-bit)
    assign m_axi_awburst = 2'b01;  // INCR
    assign m_axi_arsize  = 3'b011;
    assign m_axi_arburst = 2'b01;

    always_comb begin
        m_axi_awaddr  = '0; m_axi_awlen   = '0; m_axi_awvalid = 1'b0;
        m_axi_wdata   = '0; m_axi_wstrb   = '0; m_axi_wlast   = 1'b0; m_axi_wvalid  = 1'b0;
        m_axi_bready  = 1'b0;
        m_axi_araddr  = '0; m_axi_arlen   = '0; m_axi_arvalid = 1'b0;
        m_axi_rready  = 1'b0;

        p1_awready = 1'b0; p1_wready = 1'b0; 
        p2_arready = 1'b0; p2_rdata  = '0;   p2_rlast  = 1'b0; p2_rvalid = 1'b0;
        roi_arready= 1'b0; roi_rdata = '0;   roi_rlast = 1'b0; roi_rvalid= 1'b0;

        case (state)
            ST_PASS1_STREAM: begin
                m_axi_awaddr  = p1_awaddr; m_axi_awlen = p1_awlen; m_axi_awvalid = p1_awvalid; p1_awready = m_axi_awready;
                m_axi_wdata   = p1_wdata;  m_axi_wstrb = p1_wstrb; m_axi_wlast   = p1_wlast;   m_axi_wvalid = p1_wvalid; p1_wready = m_axi_wready;
                m_axi_bready  = p1_bready;
            end
            ST_PASS2_STATS: begin
                m_axi_araddr  = p2_araddr; m_axi_arlen = p2_arlen; m_axi_arvalid = p2_arvalid; p2_arready = m_axi_arready;
                m_axi_rready  = p2_rready; p2_rdata    = m_axi_rdata; p2_rlast   = m_axi_rlast; p2_rvalid = m_axi_rvalid;
            end
            ST_ROI_FETCH: begin
                m_axi_araddr  = roi_araddr; m_axi_arlen = roi_arlen; m_axi_arvalid = roi_arvalid; roi_arready = m_axi_arready;
                m_axi_rready  = roi_rready; roi_rdata   = m_axi_rdata; roi_rlast   = m_axi_rlast; roi_rvalid  = m_axi_rvalid;
            end
            default: ;
        endcase
    end

    // ==========================================
    // Result Latching for CSR
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latched_best_class_id <= 8'd0;
            stream_row_cnt <= '0;
            frame_eof_seen <= 1'b0;
            geo_done_seen <= 1'b0;
            geo_obj_seen <= 1'b0;
            detected_id_reg <= 8'hFF;
            bbox_valid_reg  <= 1'b0;
        end else begin
            if (state == ST_IDLE) begin
                stream_row_cnt <= '0;
                frame_eof_seen <= 1'b0;
                geo_done_seen <= 1'b0;
                geo_obj_seen <= 1'b0;
            end else begin
                if (state == ST_PASS1_STREAM && s_axis_tvalid) begin
                    if (s_axis_tuser) begin
                        stream_row_cnt <= '0;
                        frame_eof_seen <= 1'b0;
                    end
                    if (s_axis_tlast) begin
                        if (stream_row_cnt == IMG_H - 1) begin
                            frame_eof_seen <= 1'b1;
                        end else begin
                            stream_row_cnt <= stream_row_cnt + 1'b1;
                        end
                    end
                end

                if (state == ST_GEOMETRY) begin
                    if (geo_done) begin
                        geo_done_seen <= 1'b1;
                    end
                    if (obj_valid) begin
                        geo_obj_seen <= 1'b1;
                    end
                end else begin
                    geo_done_seen <= 1'b0;
                    geo_obj_seen <= 1'b0;
                end
            end

            // Sample the result ONLY when the Matcher explicitly says it's done
            if (state == ST_TEMPLATE_MACH && match_done) begin
                if (best_score < NO_SIGN_SCORE_THRESHOLD) begin
                    latched_best_class_id <= matcher_best_class_id;
                    detected_id_reg <= matcher_best_class_id;
                    bbox_valid_reg  <= 1'b1;
                end else begin
                    detected_id_reg <= 8'hFF;
                    bbox_valid_reg  <= 1'b0;
                end
            end

            // If we transition to ST_DONE from ST_GEOMETRY with no detection, force outputs to 'no detection'
            if (state == ST_GEOMETRY && next_state == ST_DONE && !(obj_valid || geo_obj_seen)) begin
                detected_id_reg <= 8'hFF;
                bbox_valid_reg  <= 1'b0;
            end
        end
    end
    
    // Connect the LATCHED value to the output port
    assign sts_best_class_id = latched_best_class_id;

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
        .p2_parent_rdata (p2_parent_rdata),
        .p1_axis_tdata   (p1_axis_tdata),
        .p1_axis_tvalid  (p1_axis_tvalid),
        .p1_axis_tlast   (p1_axis_tlast)
    );

    ccl_stats_collector #(
        .LABEL_W(LABEL_W),
        .IMG_WIDTH(IMG_W),
        .IMG_HEIGHT(IMG_H),
        .USE_RESOLVED_LABELS(1'b1)
    ) u_ccl_stats (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear           (stats_clear),
        .s_axis_label    (p2_stream_tdata), 
        .s_axis_tvalid   (p2_stream_tvalid), 
        .s_axis_tuser    (1'b0), // Prevent stats reset bug
        .s_axis_tlast    (p2_stream_tlast),
        .parent_addr     (stats_parent_addr),
        .parent_rdata    (p2_parent_rdata),
        .geo_ram_addr    (geo_ram_addr),
        .out_area        (stats_area_rdata),
        .out_perimeter   (stats_perim_rdata),
        .out_xmin        (stats_xmin_rdata),
        .out_xmax        (stats_xmax_rdata),
        .out_ymin        (stats_ymin_rdata),
        .out_ymax        (stats_ymax_rdata),
        .stats_done      (stats_done),
        .init_done       (stats_init_done)
    );

    geometry_filter #(
        .LABEL_W(LABEL_W),
        .MIN_AREA_TH(300),
        .MAX_AREA_TH(100000),
        .MIN_W_TH(34),
        .MIN_H_TH(32),
        .MIN_PIX_AREA_TH(313),
        .MIN_W_RELAX_TH(31),
        .MIN_H_RELAX_TH(30),
        .FILL_MIN_NUM(218),
        .FILL_MIN_DEN(1000),
        .RELAX_SOL_NUM(400),
        .RELAX_SOL_DEN(1000),
        .ASPECT_RELAX_NUM(22),
        .ASPECT_RELAX_DEN(10),
        .MAX_CANDIDATES(5)
    ) u_geometry_filter (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (state == ST_GEOMETRY),
        .max_label     (16'hFFFE), // Protected against overflow
        .ram_addr      (geo_ram_addr),
        .ram_area      (stats_area_rdata),
        .ram_perimeter (stats_perim_rdata),
        .ram_xmin      (stats_xmin_rdata),
        .ram_xmax      (stats_xmax_rdata),
        .ram_ymin      (stats_ymin_rdata),
        .ram_ymax      (stats_ymax_rdata),
        .filter_done   (geo_done),
        .obj_valid     (obj_valid),
        .obj_label     (best_label),
        .obj_xmin      (sts_bbox_xmin),
        .obj_xmax      (sts_bbox_xmax),
        .obj_ymin      (sts_bbox_ymin),
        .obj_ymax      (sts_bbox_ymax)
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

    template_matching_engine #() u_matcher (
        .clk                (clk),
        .rst_n              (rst_n),
        .s_axis_gray_tdata  (fetch_gray_tdata),
        .s_axis_gray_tvalid (fetch_gray_tvalid),
        .s_axis_gray_tlast  (fetch_gray_tlast),
        .template_ram_addr  (tmpl_ram_addr),
        .template_ram_rdata (tmpl_ram_rdata),
        .match_done         (match_done),
        .best_class_id      (matcher_best_class_id),
        .best_score         (best_score)
    );

    assign sts_done_flag = (state == ST_DONE);

endmodule