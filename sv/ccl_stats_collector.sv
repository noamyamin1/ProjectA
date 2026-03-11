`timescale 1ns / 1ps

module ccl_stats_collector #(
    parameter int LABEL_W = 9,
    parameter int COORD_W = 12,
    parameter int MAX_COORD = 4095,
    parameter int IMG_WIDTH = 1920,
    parameter int IMG_HEIGHT = 1080
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic [LABEL_W-1:0] s_axis_label,
    input  logic               s_axis_tvalid,
    input  logic               s_axis_tuser,
    input  logic               s_axis_tlast,

    output logic [LABEL_W-1:0] parent_addr,
    input  logic [LABEL_W-1:0] parent_rdata,

    output logic [LABEL_W-1:0] geo_ram_addr,
    output logic [31:0]        out_area,
    output logic [31:0]        out_perimeter,
    output logic [15:0]        out_xmin,
    output logic [15:0]        out_xmax,
    output logic [15:0]        out_ymin,
    output logic [15:0]        out_ymax,

    output logic               stats_done
);

    typedef enum logic [1:0] {
        ST_INIT,
        ST_PROCESS,
        ST_DONE
    } state_t;
    
    state_t state, next_state;
    logic [LABEL_W-1:0] init_addr;
    logic init_en;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_INIT;
            init_addr <= '0;
        end else begin
            case (state)
                ST_INIT: begin
                    if (init_addr == {LABEL_W{1'b1}}) state <= ST_PROCESS;
                    else                              init_addr <= init_addr + 1;
                end
                ST_PROCESS: begin
                    if (s_axis_tvalid && s_axis_tuser) state <= ST_INIT;
                    else if (stats_done_pulse)         state <= ST_DONE;
                end
                ST_DONE: begin
                    if (s_axis_tvalid && s_axis_tuser) state <= ST_INIT;
                end
            endcase
        end
    end

    assign init_en = (state == ST_INIT);

    // ==========================================
    // Line Buffers & Window Generation
    // ==========================================
    logic [LABEL_W-1:0] lb0 [0:IMG_WIDTH-1];
    logic [LABEL_W-1:0] lb1 [0:IMG_WIDTH-1];
    
    logic [COORD_W-1:0] wr_ptr;
    logic [COORD_W-1:0] rd_ptr;
    logic               window_valid;

    logic [LABEL_W-1:0] win_t, win_b, win_l, win_c, win_r;
    logic [COORD_W-1:0] x_cnt, y_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            x_cnt  <= '0;
            y_cnt  <= '0;
            window_valid <= 1'b0;
            win_l  <= '0;
            win_c  <= '0;
        end else if (state == ST_INIT) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            x_cnt  <= '0;
            y_cnt  <= '0;
            window_valid <= 1'b0;
        end else if (s_axis_tvalid && state == ST_PROCESS) begin
            lb0[wr_ptr] <= s_axis_label;
            lb1[wr_ptr] <= lb0[rd_ptr];
            
            win_b <= s_axis_label;
            win_c <= lb0[rd_ptr];
            win_t <= lb1[rd_ptr];
            win_l <= win_c;
            win_r <= lb0[(rd_ptr + 1) % IMG_WIDTH]; 

            if (wr_ptr == IMG_WIDTH - 1) wr_ptr <= '0;
            else                         wr_ptr <= wr_ptr + 1;
            
            if (rd_ptr == IMG_WIDTH - 1) rd_ptr <= '0;
            else                         rd_ptr <= rd_ptr + 1;

            if (y_cnt > 0 || x_cnt > 0) window_valid <= 1'b1;

            if (s_axis_tlast) begin
                x_cnt <= '0;
                y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end else begin
            window_valid <= 1'b0;
        end
    end

    logic stats_done_pulse;
    assign stats_done_pulse = (y_cnt == IMG_HEIGHT - 1) && s_axis_tlast && s_axis_tvalid;
    assign stats_done = (state == ST_DONE);

    // ==========================================
    // Edge Detection
    // ==========================================
    logic is_edge;
    logic valid_pixel;

    always_comb begin
        is_edge = 1'b0;
        if (win_c != 0) begin
            if ((win_c != win_t) || (win_c != win_b) || (win_c != win_l) || (win_c != win_r)) begin
                is_edge = 1'b1;
            end
            if (x_cnt == 1 || x_cnt == IMG_WIDTH - 1 || y_cnt == 1 || y_cnt == IMG_HEIGHT - 1) begin
                is_edge = 1'b1;
            end
        end
        valid_pixel = window_valid && (win_c != 0);
    end

    // ==========================================
    // RMW Pipeline
    // ==========================================
    logic               stg1_valid, stg2_valid, stg3_valid;
    logic [COORD_W-1:0] stg1_x, stg1_y, stg2_x, stg2_y, stg3_x, stg3_y;
    logic [LABEL_W-1:0] stg1_label, stg2_root, stg3_root;
    logic               stg1_edge, stg2_edge, stg3_edge;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg1_valid <= 1'b0; stg2_valid <= 1'b0; stg3_valid <= 1'b0;
        end else begin
            stg1_valid <= valid_pixel;
            stg2_valid <= stg1_valid;
            stg3_valid <= stg2_valid;
            
            if (valid_pixel) begin
                stg1_x <= x_cnt - 1; 
                stg1_y <= y_cnt > 0 ? y_cnt - 1 : '0;
                stg1_label <= win_c;
                stg1_edge  <= is_edge;
            end
            if (stg1_valid) begin
                stg2_x <= stg1_x; stg2_y <= stg1_y;
                stg2_edge <= stg1_edge;
            end
            if (stg2_valid) begin
                stg3_x <= stg2_x; stg3_y <= stg2_y;
                stg3_root <= stg2_root;
                stg3_edge <= stg2_edge;
            end
        end
    end

    assign parent_addr = stg1_label;
    assign stg2_root   = parent_rdata;

    // ==========================================
    // RAMs Definition
    // ==========================================
    logic [31:0] area_ram  [0:(1<<LABEL_W)-1];
    logic [31:0] perim_ram [0:(1<<LABEL_W)-1];
    logic [15:0] xmin_ram  [0:(1<<LABEL_W)-1];
    logic [15:0] xmax_ram  [0:(1<<LABEL_W)-1];
    logic [15:0] ymin_ram  [0:(1<<LABEL_W)-1];
    logic [15:0] ymax_ram  [0:(1<<LABEL_W)-1];

    logic [LABEL_W-1:0] ram_addr_rd, ram_addr_wr;
    logic               ram_we;
    
    logic [31:0] area_rd, area_wr, perim_rd, perim_wr;
    logic [15:0] xmin_rd, xmin_wr, xmax_rd, xmax_wr;
    logic [15:0] ymin_rd, ymin_wr, ymax_rd, ymax_wr;

    assign ram_addr_rd = stg2_root;
    assign ram_addr_wr = init_en ? init_addr : stg3_root;
    assign ram_we      = init_en | stg3_valid;

    always_ff @(posedge clk) begin
        if (ram_we) begin
            area_ram[ram_addr_wr]  <= init_en ? 32'd0 : area_wr;
            perim_ram[ram_addr_wr] <= init_en ? 32'd0 : perim_wr;
            xmin_ram[ram_addr_wr]  <= init_en ? MAX_COORD[15:0] : xmin_wr;
            xmax_ram[ram_addr_wr]  <= init_en ? 16'd0 : xmax_wr;
            ymin_ram[ram_addr_wr]  <= init_en ? MAX_COORD[15:0] : ymin_wr;
            ymax_ram[ram_addr_wr]  <= init_en ? 16'd0 : ymax_wr;
        end
        area_rd  <= area_ram[ram_addr_rd];
        perim_rd <= perim_ram[ram_addr_rd];
        xmin_rd  <= xmin_ram[ram_addr_rd];
        xmax_rd  <= xmax_ram[ram_addr_rd];
        ymin_rd  <= ymin_ram[ram_addr_rd];
        ymax_rd  <= ymax_ram[ram_addr_rd];
    end

    // ==========================================
    // Forwarding & Compute
    // ==========================================
    logic fwd_match;
    assign fwd_match = stg3_valid && (stg3_root == stg2_root);

    logic [31:0] fwd_area, fwd_perim;
    logic [15:0] fwd_xmin, fwd_xmax, fwd_ymin, fwd_ymax;

    always_comb begin
        fwd_area  = fwd_match ? area_wr  : area_rd;
        fwd_perim = fwd_match ? perim_wr : perim_rd;
        fwd_xmin  = fwd_match ? xmin_wr  : xmin_rd;
        fwd_xmax  = fwd_match ? xmax_wr  : xmax_rd;
        fwd_ymin  = fwd_match ? ymin_wr  : ymin_rd;
        fwd_ymax  = fwd_match ? ymax_wr  : ymax_rd;

        area_wr  = fwd_area + 1;
        perim_wr = stg3_edge ? (fwd_perim + 1) : fwd_perim;
        
        xmin_wr = (stg3_x < fwd_xmin) ? stg3_x : fwd_xmin;
        xmax_wr = (stg3_x > fwd_xmax) ? stg3_x : fwd_xmax;
        ymin_wr = (stg3_y < fwd_ymin) ? stg3_y : fwd_ymin;
        ymax_wr = (stg3_y > fwd_ymax) ? stg3_y : fwd_ymax;
    end

    // ==========================================
    // Readout Interface for Geometry Filter
    // ==========================================
    assign out_area      = area_rd;
    assign out_perimeter = perim_rd;
    assign out_xmin      = xmin_rd;
    assign out_xmax      = xmax_rd;
    assign out_ymin      = ymin_rd;
    assign out_ymax      = ymax_rd;

    always_comb begin
        if (state == ST_DONE) ram_addr_rd = geo_ram_addr;
        else                  ram_addr_rd = stg2_root;
    end

endmodule