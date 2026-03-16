`timescale 1ns / 1ps

module geometry_filter #(
    parameter int LABEL_W = 16,
    parameter int MIN_AREA_TH = 1000, 
    parameter int MAX_AREA_TH = 100000
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start,
    input  logic [LABEL_W-1:0] max_label,

    output logic [LABEL_W-1:0] ram_addr,
    input  logic [31:0]        ram_area,
    input  logic [31:0]        ram_perimeter, // Not used in this specific SW model, kept for port compatibility
    input  logic [15:0]        ram_xmin,
    input  logic [15:0]        ram_xmax,
    input  logic [15:0]        ram_ymin,
    input  logic [15:0]        ram_ymax,

    output logic               filter_done,
    
    // Valid output objects interface
    output logic               obj_valid,
    output logic [LABEL_W-1:0] obj_label,
    output logic [15:0]        obj_xmin,
    output logic [15:0]        obj_xmax,
    output logic [15:0]        obj_ymin,
    output logic [15:0]        obj_ymax
);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_READ_REQ,
        ST_WAIT_RD,
        ST_CALC_DIMS, // New state: Calculate width/height
        ST_CALC_EVAL, // Renamed: Evaluate conditions
        ST_DONE
    } state_t;

    state_t state, next_state;
    logic [LABEL_W-1:0] curr_label;

    logic [15:0] width;
    logic [15:0] height;
    logic [31:0] spatial_area;
    logic [31:0] pixel_mass;

    logic        pass_min_area;
    logic        pass_aspect_ratio;
    logic        pass_hollow;
    logic        is_valid_sign;

    // FSM State and Label Counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_IDLE;
            curr_label <= '0;
        end else begin
            state <= next_state;
            
            if (state == ST_IDLE && start) begin
                curr_label <= 'd1;
            end else if (state == ST_CALC_EVAL) begin
                curr_label <= curr_label + 1'b1;
            end
        end
    end

    // FSM Next State Logic
    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start && max_label > 0) next_state = ST_READ_REQ;
                else if (start)             next_state = ST_DONE;
            end
            ST_READ_REQ:  next_state = ST_WAIT_RD;
            ST_WAIT_RD:   next_state = ST_CALC_DIMS;
            ST_CALC_DIMS: next_state = ST_CALC_EVAL;
            ST_CALC_EVAL: begin
                if (curr_label == max_label) next_state = ST_DONE;
                else                         next_state = ST_READ_REQ;
            end
            ST_DONE: begin
                if (!start) next_state = ST_IDLE;
            end
            default: next_state = ST_IDLE;
        endcase
    end

    assign ram_addr = curr_label;

    // Pipeline Stage 1: Calculate Dimensions
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            width      <= '0;
            height     <= '0;
            pixel_mass <= '0;
        end else if (state == ST_WAIT_RD) begin
            width      <= ram_xmax - ram_xmin;
            height     <= ram_ymax - ram_ymin;
            pixel_mass <= ram_area; // Sample pixel mass from RAM
        end
    end

    // Pipeline Stage 2: Calculate Spatial Area
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spatial_area <= '0;
        end else if (state == ST_CALC_DIMS) begin
            spatial_area <= width * height;
        end
    end

    // Combinatorial Evaluation (Valid in ST_CALC_EVAL)
    always_comb begin
        // Area check
        if (spatial_area >= MIN_AREA_TH && spatial_area <= MAX_AREA_TH) begin
            pass_min_area = 1'b1;
        end else begin
            pass_min_area = 1'b0;
        end

        // Aspect Ratio check: ((w * 4) >= (h * 3)) && ((w * 3) <= (h * 4))
        if (((width << 2) >= (height * 3)) && ((width * 3) <= (height << 2))) begin
            pass_aspect_ratio = 1'b1;
        end else begin
            pass_aspect_ratio = 1'b0;
        end

        // Hollow Check: pixel_mass <= (spatial_area / 2)
        if (pixel_mass <= (spatial_area >> 1)) begin
            pass_hollow = 1'b1;
        end else begin
            pass_hollow = 1'b0;
        end

        // Prevent false positives on 0-sized boxes
        if (width == 0 || height == 0) begin
            is_valid_sign = 1'b0;
        end else begin
            is_valid_sign = pass_min_area && pass_aspect_ratio && pass_hollow;
        end
    end

    // Output Registration
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            obj_valid <= 1'b0;
            obj_label <= '0;
            obj_xmin  <= '0;
            obj_xmax  <= '0;
            obj_ymin  <= '0;
            obj_ymax  <= '0;
        end else begin
            obj_valid <= 1'b0;
            if (state == ST_CALC_EVAL && is_valid_sign) begin
                obj_valid <= 1'b1;
                obj_label <= curr_label;
                obj_xmin  <= ram_xmin; // Note: You might need to pipeline these if RAM data changes
                obj_xmax  <= ram_xmax;
                obj_ymin  <= ram_ymin;
                obj_ymax  <= ram_ymax;
            end
        end
    end

    assign filter_done = (state == ST_DONE);

endmodule