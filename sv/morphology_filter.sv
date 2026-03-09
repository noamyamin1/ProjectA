module morphology_filter #(
    parameter int IMG_WIDTH = 1920
)(
    input  logic clk,
    input  logic rst_n,

    input  logic s_axis_tdata,
    input  logic s_axis_tvalid,
    input  logic s_axis_tuser,
    input  logic s_axis_tlast,

    output logic m_axis_tdata,
    output logic m_axis_tvalid,
    output logic m_axis_tuser,
    output logic m_axis_tlast
);

    // ==========================================
    // Stage 1: Dilation (OR Tree)
    // ==========================================
    logic [2:0][2:0] dil_window;
    logic            dil_valid;
    logic            dil_user;
    logic            dil_last;
    logic            dil_out;

    sliding_window_3x3 #(
        .IMG_WIDTH(IMG_WIDTH)
    ) u_dil_window (
        .clk     (clk),
        .rst_n   (rst_n),
        .s_valid (s_axis_tvalid),
        .s_data  (s_axis_tdata),
        .s_user  (s_axis_tuser),
        .s_last  (s_axis_tlast),
        .m_valid (dil_valid),
        .window  (dil_window),
        .m_user  (dil_user),
        .m_last  (dil_last)
    );

    assign dil_out = dil_window[0][0] | dil_window[0][1] | dil_window[0][2] |
                     dil_window[1][0] | dil_window[1][1] | dil_window[1][2] |
                     dil_window[2][0] | dil_window[2][1] | dil_window[2][2];

    // ==========================================
    // Stage 2: Erosion (AND Tree)
    // ==========================================
    logic [2:0][2:0] ero_window;
    logic            ero_valid;
    logic            ero_user;
    logic            ero_last;
    logic            ero_out;

    sliding_window_3x3 #(
        .IMG_WIDTH(IMG_WIDTH)
    ) u_ero_window (
        .clk     (clk),
        .rst_n   (rst_n),
        .s_valid (dil_valid),
        .s_data  (dil_out),
        .s_user  (dil_user),
        .s_last  (dil_last),
        .m_valid (ero_valid),
        .window  (ero_window),
        .m_user  (ero_user),
        .m_last  (ero_last)
    );

    assign ero_out = ero_window[0][0] & ero_window[0][1] & ero_window[0][2] &
                     ero_window[1][0] & ero_window[1][1] & ero_window[1][2] &
                     ero_window[2][0] & ero_window[2][1] & ero_window[2][2];

    assign m_axis_tdata  = ero_out;
    assign m_axis_tvalid = ero_valid;
    assign m_axis_tuser  = ero_user;
    assign m_axis_tlast  = ero_last;

endmodule