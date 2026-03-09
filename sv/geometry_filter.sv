module geometry_filter #(
    parameter int LABEL_W = 9
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start,
    input  logic [LABEL_W-1:0] max_label,
    input  logic [31:0]        min_area_th,
    
    output logic [LABEL_W-1:0] ram_addr,
    input  logic [31:0]        area_rdata,
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
    logic [15:0]        stg2_w;
    logic [15:0]        stg2_h;
    logic [15:0]        stg2_xmin, stg2_xmax, stg2_ymin, stg2_ymax;

    logic               stg3_valid;
    logic [LABEL_W-1:0] stg3_label;
    logic [31:0]        stg3_area;
    logic [31:0]        stg3_bbox_area;
    logic [15:0]        stg3_xmin, stg3_xmax, stg3_ymin, stg3_ymax;

    logic [31:0]        max_area_reg;

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
                if (start && max_label > 0)
                    next_state = ST_SCAN;
                else if (start)
                    next_state = ST_DONE;
            end
            ST_SCAN: begin
                if (scan_cnt == max_label)
                    next_state = ST_FLUSH;
            end
            ST_FLUSH: begin
                if (flush_cnt == 3'd3)
                    next_state = ST_DONE;
            end
            ST_DONE: begin
                if (!start)
                    next_state = ST_IDLE;
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scan_cnt <= '0;
            flush_cnt <= '0;
        end else begin
            if (state == ST_IDLE) begin
                scan_cnt <= 9'd1;
                flush_cnt <= '0;
            end else if (state == ST_SCAN) begin
                scan_cnt <= scan_cnt + 1;
            end else if (state == ST_FLUSH) begin
                flush_cnt <= flush_cnt + 1;
            end
        end
    end

    assign ram_addr = scan_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg1_valid <= 1'b0;
            stg1_label <= '0;
        end else begin
            stg1_valid <= (state == ST_SCAN);
            stg1_label <= scan_cnt;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg2_valid <= 1'b0;
            stg2_label <= '0;
            stg2_area  <= '0;
            stg2_w     <= '0;
            stg2_h     <= '0;
            stg2_xmin  <= '0;
            stg2_xmax  <= '0;
            stg2_ymin  <= '0;
            stg2_ymax  <= '0;
        end else begin
            stg2_valid <= stg1_valid;
            stg2_label <= stg1_label;
            if (stg1_valid) begin
                stg2_area <= area_rdata;
                stg2_w    <= xmax_rdata - xmin_rdata + 16'd1;
                stg2_h    <= ymax_rdata - ymin_rdata + 16'd1;
                stg2_xmin <= xmin_rdata;
                stg2_xmax <= xmax_rdata;
                stg2_ymin <= ymin_rdata;
                stg2_ymax <= ymax_rdata;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg3_valid <= 1'b0;
            stg3_label <= '0;
            stg3_area  <= '0;
            stg3_bbox_area <= '0;
            stg3_xmin  <= '0;
            stg3_xmax  <= '0;
            stg3_ymin  <= '0;
            stg3_ymax  <= '0;
        end else begin
            stg3_valid <= stg2_valid;
            stg3_label <= stg2_label;
            if (stg2_valid) begin
                stg3_area <= stg2_area;
                stg3_bbox_area <= stg2_w * stg2_h;
                stg3_xmin <= stg2_xmin;
                stg3_xmax <= stg2_xmax;
                stg3_ymin <= stg2_ymin;
                stg3_ymax <= stg2_ymax;
            end
        end
    end

    logic valid_shape;
    logic [31:0] area_scaled;
    logic [31:0] bbox_scaled;
    
    assign area_scaled = {stg3_area[23:0], 8'd0}; 
    assign bbox_scaled = stg3_bbox_area * 8'd102; 

    assign valid_shape = (area_scaled > bbox_scaled) && (stg3_area >= min_area_th);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_area_reg <= '0;
            best_label   <= '0;
            best_xmin    <= '0;
            best_xmax    <= '0;
            best_ymin    <= '0;
            best_ymax    <= '0;
        end else begin
            if (state == ST_IDLE) begin
                max_area_reg <= '0;
                best_label   <= '0;
            end else if (stg3_valid && valid_shape) begin
                if (stg3_area > max_area_reg) begin
                    max_area_reg <= stg3_area;
                    best_label   <= stg3_label;
                    best_xmin    <= stg3_xmin;
                    best_xmax    <= stg3_xmax;
                    best_ymin    <= stg3_ymin;
                    best_ymax    <= stg3_ymax;
                end
            end
        end
    end

    assign done = (state == ST_DONE);

endmodule