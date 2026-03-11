`timescale 1ns / 1ps

module geometry_filter #(
    parameter int LABEL_W = 9
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start,
    input  logic [LABEL_W-1:0] max_label,
    input  logic [31:0]        min_area_th,
    input  logic [7:0]         circ_th_percent, 
    
    output logic [LABEL_W-1:0] ram_addr,
    input  logic [31:0]        area_rdata,
    input  logic [31:0]        perimeter_rdata,
    input  logic [15:0]        xmin_rdata,
    input  logic [15:0]        xmax_rdata,
    input  logic [15:0]        ymin_rdata,
    input  logic [15:0]        ymax_rdata,

    output logic               done,
    output logic [LABEL_W-1:0] best_label,
    output logic [15:0]        best_xmin,
    output logic [15:0]        best_xmax,
    output logic [15:0]        best_ymin,
    output logic [15:0]        best_ymax
);

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_SCAN,
        ST_FLUSH,
        ST_DONE
    } state_e;

    state_e state, next_state;
    
    logic [LABEL_W-1:0] scan_cnt;
    logic [2:0]         flush_cnt;

    logic               stg1_valid;
    logic [LABEL_W-1:0] stg1_label;
    
    logic               stg2_valid;
    logic [LABEL_W-1:0] stg2_label;
    logic [31:0]        stg2_area;
    logic [31:0]        stg2_perimeter;
    logic [15:0]        stg2_xmin, stg2_xmax, stg2_ymin, stg2_ymax;

    logic               stg3_valid;
    logic [LABEL_W-1:0] stg3_label;
    logic [31:0]        stg3_area;
    logic [63:0]        stg3_perim_sq; 
    logic [63:0]        stg3_area_scaled;
    logic [15:0]        stg3_xmin, stg3_xmax, stg3_ymin, stg3_ymax;

    logic               stg4_valid;
    logic [LABEL_W-1:0] stg4_label;
    logic [31:0]        stg4_area;
    logic [63:0]        stg4_area_scaled;
    logic [63:0]        stg4_circ_thresh;
    logic [15:0]        stg4_xmin, stg4_xmax, stg4_ymin, stg4_ymax;

    logic [31:0]        max_area_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= ST_IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start && max_label > 0) next_state = ST_SCAN;
                else if (start)             next_state = ST_DONE;
            end
            ST_SCAN: begin
                if (scan_cnt == max_label)  next_state = ST_FLUSH;
            end
            ST_FLUSH: begin
                if (flush_cnt == 3'd4)      next_state = ST_DONE;
            end
            ST_DONE: begin
                if (!start)                 next_state = ST_IDLE;
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scan_cnt  <= '0;
            flush_cnt <= '0;
        end else begin
            if (state == ST_IDLE) begin
                scan_cnt  <= 9'd1;
                flush_cnt <= '0;
            end else if (state == ST_SCAN) begin
                scan_cnt  <= scan_cnt + 1;
            end else if (state == ST_FLUSH) begin
                flush_cnt <= flush_cnt + 1;
            end
        end
    end

    assign ram_addr = scan_cnt;

    // Stage 1: Address Generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg1_valid <= 1'b0;
            stg1_label <= '0;
        end else begin
            stg1_valid <= (state == ST_SCAN);
            stg1_label <= scan_cnt;
        end
    end

    // Stage 2: Memory Read
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg2_valid     <= 1'b0;
            stg2_label     <= '0;
            stg2_area      <= '0;
            stg2_perimeter <= '0;
            stg2_xmin <= '0; stg2_xmax <= '0;
            stg2_ymin <= '0; stg2_ymax <= '0;
        end else begin
            stg2_valid <= stg1_valid;
            stg2_label <= stg1_label;
            if (stg1_valid) begin
                stg2_area      <= area_rdata;
                stg2_perimeter <= perimeter_rdata;
                stg2_xmin <= xmin_rdata; stg2_xmax <= xmax_rdata;
                stg2_ymin <= ymin_rdata; stg2_ymax <= ymax_rdata;
            end
        end
    end

    // Stage 3: Initial Multiplications (Perimeter^2 and Area * 1256)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg3_valid       <= 1'b0;
            stg3_label       <= '0;
            stg3_area        <= '0;
            stg3_perim_sq    <= '0;
            stg3_area_scaled <= '0;
            stg3_xmin <= '0; stg3_xmax <= '0;
            stg3_ymin <= '0; stg3_ymax <= '0;
        end else begin
            stg3_valid <= stg2_valid;
            stg3_label <= stg2_label;
            if (stg2_valid) begin
                stg3_area        <= stg2_area;
                stg3_perim_sq    <= stg2_perimeter * stg2_perimeter;
                stg3_area_scaled <= stg2_area * 32'd1256;
                stg3_xmin <= stg2_xmin; stg3_xmax <= stg2_xmax;
                stg3_ymin <= stg2_ymin; stg3_ymax <= stg2_ymax;
            end
        end
    end

    // Stage 4: Threshold Multiplication
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg4_valid       <= 1'b0;
            stg4_label       <= '0;
            stg4_area        <= '0;
            stg4_area_scaled <= '0;
            stg4_circ_thresh <= '0;
            stg4_xmin <= '0; stg4_xmax <= '0;
            stg4_ymin <= '0; stg4_ymax <= '0;
        end else begin
            stg4_valid <= stg3_valid;
            stg4_label <= stg3_label;
            if (stg3_valid) begin
                stg4_area        <= stg3_area;
                stg4_area_scaled <= stg3_area_scaled;
                stg4_circ_thresh <= stg3_perim_sq * circ_th_percent;
                stg4_xmin <= stg3_xmin; stg4_xmax <= stg3_xmax;
                stg4_ymin <= stg3_ymin; stg4_ymax <= stg3_ymax;
            end
        end
    end

    // Stage 5: Comparison and Max Search
    logic valid_shape;
    assign valid_shape = (stg4_area_scaled > stg4_circ_thresh) && (stg4_area >= min_area_th);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_area_reg <= '0;
            best_label   <= '0;
            best_xmin <= '0; best_xmax <= '0;
            best_ymin <= '0; best_ymax <= '0;
        end else begin
            if (state == ST_IDLE) begin
                max_area_reg <= '0;
                best_label   <= '0;
            end else if (stg4_valid && valid_shape) begin
                if (stg4_area > max_area_reg) begin
                    max_area_reg <= stg4_area;
                    best_label   <= stg4_label;
                    best_xmin    <= stg4_xmin;
                    best_xmax    <= stg4_xmax;
                    best_ymin    <= stg4_ymin;
                    best_ymax    <= stg4_ymax;
                end
            end
        end
    end

    assign done = (state == ST_DONE);

endmodule