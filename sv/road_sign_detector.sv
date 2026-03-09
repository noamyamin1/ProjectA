module road_sign_detector_top #(
    parameter int AXI_LITE_ADDR_W = 12,
    parameter int AXI_LITE_DATA_W = 32,
    parameter int AXIS_TDATA_W    = 24,
    parameter int AXI_FULL_ADDR_W = 32,
    parameter int AXI_FULL_DATA_W = 64,
    parameter int IMG_W           = 1920,
    parameter int IMG_H           = 1080
)(
    input  logic clk,
    input  logic rst_n,

    // AXI4-Lite Slave Interface (Control & Status)
    input  logic [AXI_LITE_ADDR_W-1:0] s_axi_awaddr,
    input  logic                       s_axi_awvalid,
    output logic                       s_axi_awready,
    input  logic [AXI_LITE_DATA_W-1:0] s_axi_wdata,
    input  logic [3:0]                 s_axi_wstrb,
    input  logic                       s_axi_wvalid,
    output logic                       s_axi_wready,
    output logic [1:0]                 s_axi_bresp,
    output logic                       s_axi_bvalid,
    input  logic                       s_axi_bready,
    input  logic [AXI_LITE_ADDR_W-1:0] s_axi_araddr,
    input  logic                       s_axi_arvalid,
    output logic                       s_axi_arready,
    output logic [AXI_LITE_DATA_W-1:0] s_axi_rdata,
    output logic [1:0]                 s_axi_rresp,
    output logic                       s_axi_rvalid,
    input  logic                       s_axi_rready,

    // AXI4-Stream Slave Interface (Raw RGB Video Input)
    input  logic [AXIS_TDATA_W-1:0]    s_axis_tdata,
    input  logic                       s_axis_tvalid,
    output logic                       s_axis_tready,
    input  logic                       s_axis_tuser,
    input  logic                       s_axis_tlast,

    // AXI4 Master 0: Dedicated RGB Frame Writer
    output logic [AXI_FULL_ADDR_W-1:0] m0_axi_awaddr,
    output logic [7:0]                 m0_axi_awlen,
    output logic [2:0]                 m0_axi_awsize,
    output logic [1:0]                 m0_axi_awburst,
    output logic                       m0_axi_awvalid,
    input  logic                       m0_axi_awready,
    output logic [AXI_FULL_DATA_W-1:0] m0_axi_wdata,
    output logic [7:0]                 m0_axi_wstrb,
    output logic                       m0_axi_wlast,
    output logic                       m0_axi_wvalid,
    input  logic                       m0_axi_wready,
    input  logic [1:0]                 m0_axi_bresp,
    input  logic                       m0_axi_bvalid,
    output logic                       m0_axi_bready,

    // AXI4 Master 1: Backend Processing (CCL & ROI Fetch)
    output logic [AXI_FULL_ADDR_W-1:0] m1_axi_awaddr,
    output logic [7:0]                 m1_axi_awlen,
    output logic [2:0]                 m1_axi_awsize,
    output logic [1:0]                 m1_axi_awburst,
    output logic                       m1_axi_awvalid,
    input  logic                       m1_axi_awready,
    output logic [AXI_FULL_DATA_W-1:0] m1_axi_wdata,
    output logic [7:0]                 m1_axi_wstrb,
    output logic                       m1_axi_wlast,
    output logic                       m1_axi_wvalid,
    input  logic                       m1_axi_wready,
    input  logic [1:0]                 m1_axi_bresp,
    input  logic                       m1_axi_bvalid,
    output logic                       m1_axi_bready,
    output logic [AXI_FULL_ADDR_W-1:0] m1_axi_araddr,
    output logic [7:0]                 m1_axi_arlen,
    output logic [2:0]                 m1_axi_arsize,
    output logic [1:0]                 m1_axi_arburst,
    output logic                       m1_axi_arvalid,
    input  logic                       m1_axi_arready,
    input  logic [AXI_FULL_DATA_W-1:0] m1_axi_rdata,
    input  logic                       m1_axi_rlast,
    input  logic [1:0]                 m1_axi_rresp,
    input  logic                       m1_axi_rvalid,
    output logic                       m1_axi_rready,

    output logic                       irq
);

    logic [7:0]  cfg_min_red_val;
    logic [2:0]  cfg_margin_shift;
    logic [31:0] cfg_frame_base_addr;
    logic        cfg_enable;
    
    logic        sts_done_flag;
    logic [7:0]  sts_best_class_id;
    logic [15:0] sts_bbox_xmin;
    logic [15:0] sts_bbox_xmax;
    logic [15:0] sts_bbox_ymin;
    logic [15:0] sts_bbox_ymax;

    logic        stream_mask_tdata;
    logic        stream_mask_tvalid;
    logic        stream_mask_tuser;
    logic        stream_mask_tlast;
    
    logic        stream_morph_tdata;
    logic        stream_morph_tvalid;
    logic        stream_morph_tuser;
    logic        stream_morph_tlast;

    csr_unit #(
        .ADDR_W(AXI_LITE_ADDR_W),
        .DATA_W(AXI_LITE_DATA_W)
    ) u_csr_unit (
        .clk                 (clk),
        .rst_n               (rst_n),
        .s_axi_awaddr        (s_axi_awaddr),
        .s_axi_awvalid       (s_axi_awvalid),
        .s_axi_awready       (s_axi_awready),
        .s_axi_wdata         (s_axi_wdata),
        .s_axi_wstrb         (s_axi_wstrb),
        .s_axi_wvalid        (s_axi_wvalid),
        .s_axi_wready        (s_axi_wready),
        .s_axi_bresp         (s_axi_bresp),
        .s_axi_bvalid        (s_axi_bvalid),
        .s_axi_bready        (s_axi_bready),
        .s_axi_araddr        (s_axi_araddr),
        .s_axi_arvalid       (s_axi_arvalid),
        .s_axi_arready       (s_axi_arready),
        .s_axi_rdata         (s_axi_rdata),
        .s_axi_rresp         (s_axi_rresp),
        .s_axi_rvalid        (s_axi_rvalid),
        .s_axi_rready        (s_axi_rready),
        .cfg_min_red_val     (cfg_min_red_val),
        .cfg_margin_shift    (cfg_margin_shift),
        .cfg_frame_base_addr (cfg_frame_base_addr),
        .cfg_enable          (cfg_enable),
        .sts_done_flag       (sts_done_flag),
        .sts_best_class_id   (sts_best_class_id),
        .sts_bbox_xmin       (sts_bbox_xmin),
        .sts_bbox_xmax       (sts_bbox_xmax),
        .sts_bbox_ymin       (sts_bbox_ymin),
        .sts_bbox_ymax       (sts_bbox_ymax),
        .irq                 (irq)
    );

    rgb_frame_writer #(
        .AXI_ADDR_W(AXI_FULL_ADDR_W),
        .AXI_DATA_W(AXI_FULL_DATA_W)
    ) u_rgb_writer (
        .clk                 (clk),
        .rst_n               (rst_n),
        .enable              (cfg_enable),
        .base_addr           (cfg_frame_base_addr),
        .s_axis_tdata        (s_axis_tdata),
        .s_axis_tvalid       (s_axis_tvalid),
        .s_axis_tuser        (s_axis_tuser),
        .s_axis_tlast        (s_axis_tlast),
        .s_axis_tready       (s_axis_tready),
        .m_axi_awaddr        (m0_axi_awaddr),
        .m_axi_awlen         (m0_axi_awlen),
        .m_axi_awsize        (m0_axi_awsize),
        .m_axi_awburst       (m0_axi_awburst),
        .m_axi_awvalid       (m0_axi_awvalid),
        .m_axi_awready       (m0_axi_awready),
        .m_axi_wdata         (m0_axi_wdata),
        .m_axi_wstrb         (m0_axi_wstrb),
        .m_axi_wlast         (m0_axi_wlast),
        .m_axi_wvalid        (m0_axi_wvalid),
        .m_axi_wready        (m0_axi_wready),
        .m_axi_bresp         (m0_axi_bresp),
        .m_axi_bvalid        (m0_axi_bvalid),
        .m_axi_bready        (m0_axi_bready)
    );

    red_mask_datapath #(
        .TDATA_W(AXIS_TDATA_W)
    ) u_red_mask (
        .clk                 (clk),
        .rst_n               (rst_n),
        .min_red_val         (cfg_min_red_val),
        .margin_shift        (cfg_margin_shift),
        .s_axis_tdata        (s_axis_tdata),
        .s_axis_tvalid       (s_axis_tvalid),
        .s_axis_tready       (),
        .s_axis_tuser        (s_axis_tuser),
        .s_axis_tlast        (s_axis_tlast),
        .m_axis_tdata        (stream_mask_tdata),
        .m_axis_tvalid       (stream_mask_tvalid),
        .m_axis_tuser        (stream_mask_tuser),
        .m_axis_tlast        (stream_mask_tlast)
    );

    morphology_filter u_morphology (
        .clk                 (clk),
        .rst_n               (rst_n),
        .s_axis_tdata        (stream_mask_tdata),
        .s_axis_tvalid       (stream_mask_tvalid),
        .s_axis_tuser        (stream_mask_tuser),
        .s_axis_tlast        (stream_mask_tlast),
        .m_axis_tdata        (stream_morph_tdata),
        .m_axis_tvalid       (stream_morph_tvalid),
        .m_axis_tuser        (stream_morph_tuser),
        .m_axis_tlast        (stream_morph_tlast)
    );

    backend_processing_unit #(
        .AXI_ADDR_W(AXI_FULL_ADDR_W),
        .AXI_DATA_W(AXI_FULL_DATA_W),
        .IMG_W(IMG_W),
        .IMG_H(IMG_H)
    ) u_backend_processing (
        .clk                 (clk),
        .rst_n               (rst_n),
        .cfg_enable          (cfg_enable),
        .cfg_frame_base_addr (cfg_frame_base_addr),
        .s_axis_tdata        (stream_morph_tdata),
        .s_axis_tvalid       (stream_morph_tvalid),
        .s_axis_tuser        (stream_morph_tuser),
        .s_axis_tlast        (stream_morph_tlast),
        .m_axi_awaddr        (m1_axi_awaddr),
        .m_axi_awlen         (m1_axi_awlen),
        .m_axi_awsize        (m1_axi_awsize),
        .m_axi_awburst       (m1_axi_awburst),
        .m_axi_awvalid       (m1_axi_awvalid),
        .m_axi_awready       (m1_axi_awready),
        .m_axi_wdata         (m1_axi_wdata),
        .m_axi_wstrb         (m1_axi_wstrb),
        .m_axi_wlast         (m1_axi_wlast),
        .m_axi_wvalid        (m1_axi_wvalid),
        .m_axi_wready        (m1_axi_wready),
        .m_axi_bresp         (m1_axi_bresp),
        .m_axi_bvalid        (m1_axi_bvalid),
        .m_axi_bready        (m1_axi_bready),
        .m_axi_araddr        (m1_axi_araddr),
        .m_axi_arlen         (m1_axi_arlen),
        .m_axi_arsize        (m1_axi_arsize),
        .m_axi_arburst       (m1_axi_arburst),
        .m_axi_arvalid       (m1_axi_arvalid),
        .m_axi_arready       (m1_axi_arready),
        .m_axi_rdata         (m1_axi_rdata),
        .m_axi_rlast         (m1_axi_rlast),
        .m_axi_rresp         (m1_axi_rresp),
        .m_axi_rvalid        (m1_axi_rvalid),
        .m_axi_rready        (m1_axi_rready),
        .sts_done_flag       (sts_done_flag),
        .sts_best_class_id   (sts_best_class_id),
        .sts_bbox_xmin       (sts_bbox_xmin),
        .sts_bbox_xmax       (sts_bbox_xmax),
        .sts_bbox_ymin       (sts_bbox_ymin),
        .sts_bbox_ymax       (sts_bbox_ymax)
    );

endmodule