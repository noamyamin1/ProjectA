`timescale 1ns / 1ps

module geometry_filter #(
    parameter int LABEL_W = 16,
    parameter int MIN_AREA_TH = 1000, 
    parameter int MAX_AREA_TH = 100000,
    parameter int MIN_W_TH = 32,
    parameter int MIN_H_TH = 32,
    parameter int MIN_PIX_AREA_TH = 313,
    parameter int MIN_W_RELAX_TH = 31,
    parameter int MIN_H_RELAX_TH = 30,
    parameter int CIRC_MIN_NUM = 12566,
    parameter int CIRC_MIN_DEN = 100,
    parameter int FILL_MIN_NUM = 217,
    parameter int FILL_MIN_DEN = 1000,
    parameter int RELAX_SOL_NUM = 400,
    parameter int RELAX_SOL_DEN = 1000,
    parameter int FILL_MAX_NUM = 60,
    parameter int FILL_MAX_DEN = 100,
    parameter int ASPECT_RELAX_NUM = 22,
    parameter int ASPECT_RELAX_DEN = 10,
    parameter int MAX_CANDIDATES = 5
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
    logic [31:0] perim_mass;
    logic [63:0] perim_sq;

    logic        pass_min_area;
    logic        pass_min_dims;
    logic        pass_min_dims_relax;
    logic        pass_min_dims_final;
    logic        pass_min_pix_area;
    logic        pass_aspect_ratio;
    logic        pass_aspect_relax;
    logic        pass_aspect_final;
    logic        pass_circularity;
    logic        pass_fill_min;
    logic        pass_fill_ratio;
    logic        is_valid_sign;
    logic        pass_candidate;
    logic [95:0] circ_lhs;
    logic [95:0] circ_rhs;
    logic [63:0] fill_lhs;
    logic [63:0] fill_rhs;
    logic [63:0] fill_min_lhs;
    logic [63:0] fill_min_rhs;

    logic [31:0] best_area;
    logic [31:0] best_spatial_area;
    logic [63:0] best_perim_sq;
    logic [15:0] best_delta;
    logic        best_valid;
    logic        obj_sent;
    logic [LABEL_W-1:0] best_label;
    logic [15:0]        best_xmin;
    logic [15:0]        best_xmax;
    logic [15:0]        best_ymin;
    logic [15:0]        best_ymax;
    logic [15:0]        pass_count;
    logic               too_many;
    logic [15:0]        cand_delta;
    logic [95:0]        cand_score_lhs;
    logic [95:0]        cand_score_rhs;
    logic [63:0]        cand_fill_lhs;
    logic [63:0]        cand_fill_rhs;

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
            perim_mass <= '0;
        end else if (state == ST_WAIT_RD) begin
            width      <= ram_xmax - ram_xmin + 1;
            height     <= ram_ymax - ram_ymin + 1;
            pixel_mass <= ram_area; // Sample pixel mass from RAM
            perim_mass <= ram_perimeter;
        end
    end

    // Pipeline Stage 2: Calculate Spatial Area
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spatial_area <= '0;
            perim_sq <= '0;
        end else if (state == ST_CALC_DIMS) begin
            spatial_area <= width * height;
            perim_sq <= perim_mass * perim_mass;
        end
    end

    // Combinatorial Evaluation (Valid in ST_CALC_EVAL)
    always_comb begin
        // Area check (spatial area)
        if (spatial_area >= MIN_AREA_TH && spatial_area <= MAX_AREA_TH) begin
            pass_min_area = 1'b1;
        end else begin
            pass_min_area = 1'b0;
        end

        // Min width/height check
        if (width >= MIN_W_TH && height >= MIN_H_TH) begin
            pass_min_dims = 1'b1;
        end else begin
            pass_min_dims = 1'b0;
        end

        // Relaxed dims for circular, solid candidates
        if (pass_circularity && (pixel_mass * RELAX_SOL_DEN >= spatial_area * RELAX_SOL_NUM) &&
            width >= MIN_W_RELAX_TH && height >= MIN_H_RELAX_TH) begin
            pass_min_dims_relax = 1'b1;
        end else begin
            pass_min_dims_relax = 1'b0;
        end

        pass_min_dims_final = pass_min_dims || pass_min_dims_relax;

        // Min pixel area check
        if (pixel_mass >= MIN_PIX_AREA_TH) begin
            pass_min_pix_area = 1'b1;
        end else begin
            pass_min_pix_area = 1'b0;
        end

        // Aspect Ratio check: ((w * 4) >= (h * 3)) && ((w * 3) <= (h * 4))
        if (((width << 2) >= (height * 3)) && ((width * 3) <= (height << 2))) begin
            pass_aspect_ratio = 1'b1;
        end else begin
            pass_aspect_ratio = 1'b0;
        end

        // Relaxed aspect ratio for circular + solid candidates (<= 2.2:1)
        if (pass_circularity && pass_fill_min &&
            (width * ASPECT_RELAX_DEN <= height * ASPECT_RELAX_NUM) &&
            (height * ASPECT_RELAX_DEN <= width * ASPECT_RELAX_NUM)) begin
            pass_aspect_relax = 1'b1;
        end else begin
            pass_aspect_relax = 1'b0;
        end

        pass_aspect_final = pass_aspect_ratio || pass_aspect_relax;

        circ_lhs = CIRC_MIN_NUM * pixel_mass;
        circ_rhs = CIRC_MIN_DEN * perim_sq;

        // Circularity check: 4*pi*area >= circ_min * perimeter^2
        // Implemented as: CIRC_MIN_NUM * area >= CIRC_MIN_DEN * perim_sq
        if (circ_lhs >= circ_rhs) begin
            pass_circularity = 1'b1;
        end else begin
            pass_circularity = 1'b0;
        end

        // Fill ratio check (max): area / spatial_area <= FILL_MAX_NUM / FILL_MAX_DEN
        fill_lhs = pixel_mass * FILL_MAX_DEN;
        fill_rhs = spatial_area * FILL_MAX_NUM;
        if (spatial_area == 0) begin
            pass_fill_ratio = 1'b0;
        end else if (fill_lhs <= fill_rhs) begin
            pass_fill_ratio = 1'b1;
        end else begin
            pass_fill_ratio = 1'b0;
        end

        // Fill ratio check (min): area / spatial_area >= FILL_MIN_NUM / FILL_MIN_DEN
        fill_min_lhs = pixel_mass * FILL_MIN_DEN;
        fill_min_rhs = spatial_area * FILL_MIN_NUM;
        if (spatial_area == 0) begin
            pass_fill_min = 1'b0;
        end else if (fill_min_lhs >= fill_min_rhs) begin
            pass_fill_min = 1'b1;
        end else begin
            pass_fill_min = 1'b0;
        end

        // Prevent false positives on 0-sized boxes
        pass_candidate = pass_min_area && pass_min_dims_final && pass_min_pix_area;

        if (width == 0 || height == 0) begin
            is_valid_sign = 1'b0;
        end else begin
            is_valid_sign = pass_min_area && pass_min_dims_final && pass_min_pix_area &&
                            pass_aspect_final && pass_circularity &&
                            pass_fill_ratio && pass_fill_min;
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
            best_area <= 32'd0;
            best_spatial_area <= 32'd0;
            best_perim_sq <= 64'd0;
            best_delta <= 16'hFFFF;
            best_valid <= 1'b0;
            obj_sent <= 1'b0;
            best_label <= '0;
            best_xmin <= '0;
            best_xmax <= '0;
            best_ymin <= '0;
            best_ymax <= '0;
            pass_count <= '0;
            too_many <= 1'b0;
        end else begin
            obj_valid <= 1'b0;

            if (state == ST_IDLE && start) begin
                best_area <= 32'd0;
                best_spatial_area <= 32'd0;
                best_perim_sq <= 64'd0;
                best_delta <= 16'hFFFF;
                best_valid <= 1'b0;
                obj_sent <= 1'b0;
                pass_count <= '0;
                too_many <= 1'b0;
            end

            if (state == ST_CALC_EVAL) begin
                if (is_valid_sign) begin
                    pass_count <= pass_count + 1'b1;
                    if (pass_count + 1'b1 > MAX_CANDIDATES) begin
                        too_many <= 1'b1;
                    end
                    cand_delta = (width >= height) ? (width - height) : (height - width);

                    if (!best_valid) begin
                        best_area <= pixel_mass;
                        best_spatial_area <= spatial_area;
                        best_perim_sq <= perim_sq;
                        best_delta <= cand_delta;
                        best_valid <= 1'b1;
                        best_label <= curr_label;
                        best_xmin  <= ram_xmin;
                        best_xmax  <= ram_xmax;
                        best_ymin  <= ram_ymin;
                        best_ymax  <= ram_ymax;
                    end else begin
                        cand_fill_lhs = pixel_mass * best_spatial_area;
                        cand_fill_rhs = best_area * spatial_area;
                        cand_score_lhs = pixel_mass * best_perim_sq;
                        cand_score_rhs = best_area * perim_sq;
                        if (cand_fill_lhs < cand_fill_rhs ||
                            (cand_fill_lhs == cand_fill_rhs && cand_score_lhs > cand_score_rhs) ||
                            (cand_fill_lhs == cand_fill_rhs && cand_score_lhs == cand_score_rhs && cand_delta < best_delta) ||
                            (cand_fill_lhs == cand_fill_rhs && cand_score_lhs == cand_score_rhs && cand_delta == best_delta && pixel_mass > best_area)) begin
                            best_area <= pixel_mass;
                            best_spatial_area <= spatial_area;
                            best_perim_sq <= perim_sq;
                            best_delta <= cand_delta;
                            best_valid <= 1'b1;
                            best_label <= curr_label;
                            best_xmin  <= ram_xmin;
                            best_xmax  <= ram_xmax;
                            best_ymin  <= ram_ymin;
                            best_ymax  <= ram_ymax;
                        end
                    end
                end
            end

            if (state == ST_DONE && best_valid && !obj_sent && !too_many) begin
                obj_valid <= 1'b1;
                obj_label <= best_label;
                obj_xmin  <= best_xmin;
                obj_xmax  <= best_xmax;
                obj_ymin  <= best_ymin;
                obj_ymax  <= best_ymax;
                obj_sent <= 1'b1;
            end
        end
    end

    assign filter_done = (state == ST_DONE);

endmodule