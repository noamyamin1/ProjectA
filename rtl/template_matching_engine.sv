`timescale 1ns / 1ps

module template_matching_engine #(
    parameter int TARGET_W = 64,
    parameter int TARGET_H = 64,
    parameter int TEMPLATE_COUNT = 19,
    parameter int TEMPLATE_ADDR_W = 11 
)(
    input  logic         clk,
    input  logic         rst_n,

    input  logic [7:0]   s_axis_gray_tdata,
    input  logic         s_axis_gray_tvalid,
    input  logic         s_axis_gray_tlast,
    
    output logic [TEMPLATE_ADDR_W-1:0] template_ram_addr,
    input  logic [31:0]  template_ram_rdata,

    output logic         match_done,
    output logic [7:0]   best_class_id,
    output logic [31:0]  best_score
);

    typedef enum logic [2:0] {
        ST_IDLE      = 3'b000,
        ST_RCV_ROI   = 3'b001,
        ST_CALC_MEAN = 3'b010,
        ST_BINARIZE  = 3'b011,
        ST_MATCHING  = 3'b100,
        ST_DONE      = 3'b101
    } state_e;

    state_e state, next_state;

    logic [11:0] pixel_cnt;
    logic [19:0] roi_sum;
    logic [7:0]  roi_mean;
    
    logic [7:0]  roi_buf [0:4095];
    logic [11:0] buf_rd_addr;
    logic [11:0] buf_rd_addr_d1;
    logic [7:0]  buf_rd_data;
    
    logic [31:0] bin_roi [0:31]; 
    logic [4:0]  bin_row_cnt;
    logic [5:0]  bin_col_cnt;

    logic [7:0]  template_idx;
    logic [5:0]  match_row_cnt; 
    
    logic [3:0]  dx_idx;
    logic [3:0]  dy_idx;

    logic [31:0] current_mismatches;
    logic [31:0] min_mismatches;
    logic [7:0]  min_class_id;

    // --- Binarization Logic ---
    logic [5:0] binarize_x_pos;
    logic [7:0] binarize_threshold;
    logic       binarize_bit_val;

    assign binarize_x_pos = buf_rd_addr_d1[5:0];
    assign binarize_threshold = (roi_mean > 8'd15) ? (roi_mean - 8'd15) : 8'd0;
    assign binarize_bit_val = (buf_rd_data < binarize_threshold) ? 1'b1 : 1'b0;

    // --- 2D Shift Engine Logic ---
    logic signed [5:0] actual_y;
    logic [31:0]       raw_row;
    logic signed [4:0] actual_x;

    assign actual_y = $signed({1'b0, match_row_cnt[5:0]}) + $signed({2'b00, dy_idx}) - 6'sd4;
    assign raw_row  = (actual_y >= 0 && actual_y < 32 && match_row_cnt < 32) ? bin_roi[actual_y] : 32'b0;
    assign actual_x = $signed({1'b0, dx_idx}) - 5'sd4;

    // --- Pipeline Registers ---
    logic [31:0] pl_raw_roi_row;
    logic [31:0] pl_shifted_roi_row;
    logic [31:0] pl_rom_data;
    logic [31:0] pl_valid_mask;
    
    logic pl_eval_valid_1;
    logic pl_eval_valid_2;
    logic eval_valid;

    logic [31:0] xor_result;
    logic [5:0]  popcount_val;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= ST_IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE:      if (s_axis_gray_tvalid) next_state = ST_RCV_ROI;
            ST_RCV_ROI:   if (s_axis_gray_tvalid && s_axis_gray_tlast) next_state = ST_CALC_MEAN;
            ST_CALC_MEAN: next_state = ST_BINARIZE;
            ST_BINARIZE:  if (buf_rd_addr_d1 == 4095) next_state = ST_MATCHING;
            ST_MATCHING:  if (template_idx == TEMPLATE_COUNT) next_state = ST_DONE;
            ST_DONE:      next_state = ST_IDLE;
            default:      next_state = ST_IDLE;
        endcase
    end

    // --- Input ROI Buffer & Mean ---
    always_ff @(posedge clk) begin
        if ((state == ST_IDLE || state == ST_RCV_ROI) && s_axis_gray_tvalid) begin
            roi_buf[pixel_cnt] <= s_axis_gray_tdata;
        end
        buf_rd_data <= roi_buf[buf_rd_addr];
        buf_rd_addr_d1 <= buf_rd_addr;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_cnt <= '0;
            roi_sum   <= '0;
        end else begin
            if (state == ST_IDLE) begin
                if (s_axis_gray_tvalid) begin
                    pixel_cnt <= 12'd1;
                    roi_sum   <= '0;   
                end else begin
                    pixel_cnt <= '0;
                    roi_sum   <= '0;
                end
            end else if (state == ST_RCV_ROI && s_axis_gray_tvalid) begin
                pixel_cnt <= pixel_cnt + 1;
                
                if (pixel_cnt[11:6] >= 16 && pixel_cnt[11:6] < 48 &&
                    pixel_cnt[5:0] >= 16 && pixel_cnt[5:0] < 48) begin
                    roi_sum <= roi_sum + s_axis_gray_tdata;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) roi_mean <= '0;
        else if (state == ST_CALC_MEAN) roi_mean <= roi_sum[17:10]; 
    end

    // --- Binarization Stage ---
    logic [5:0] binarize_y_pos;
    assign binarize_y_pos = buf_rd_addr_d1[11:6];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_rd_addr <= '0;
            bin_row_cnt <= '0;
            bin_col_cnt <= '0;
        end else begin
            if (state == ST_CALC_MEAN) begin
                buf_rd_addr <= '0;
                bin_row_cnt <= '0;
                bin_col_cnt <= '0;
            end else if (state == ST_BINARIZE) begin
                if (buf_rd_addr < 4095) buf_rd_addr <= buf_rd_addr + 1;

                if (binarize_y_pos >= 16 && binarize_y_pos < 48 && 
                    binarize_x_pos >= 16 && binarize_x_pos < 48) begin
                    
                    bin_roi[bin_row_cnt][5'd31 - bin_col_cnt] <= binarize_bit_val;
                    
                    if (bin_col_cnt == 5'd31) begin
                        bin_col_cnt <= '0;
                        bin_row_cnt <= bin_row_cnt + 1;
                    end else begin
                        bin_col_cnt <= bin_col_cnt + 1;
                    end
                end
            end
        end
    end

    // --- Pipeline Stage 1: Fetch Row & Calc Mask ---
    logic [31:0] current_x_mask;
    
    always_comb begin
        if (actual_x > 0)      current_x_mask = 32'hFFFF_FFFF << actual_x;
        else if (actual_x < 0) current_x_mask = 32'hFFFF_FFFF >> (-actual_x);
        else                   current_x_mask = 32'hFFFF_FFFF;
        
        if (actual_y < 0 || actual_y >= 32 || match_row_cnt >= 32) begin
            current_x_mask = 32'h0000_0000;
        end
    end

    always_ff @(posedge clk) begin
        if (state == ST_MATCHING) begin
            pl_raw_roi_row <= raw_row;
        end
    end

    // --- Pipeline Stage 2: Shift Row, Capture ROM & Mask ---
    always_ff @(posedge clk) begin
        if (state == ST_MATCHING) begin
            if (actual_x > 0)
                pl_shifted_roi_row <= pl_raw_roi_row << actual_x;
            else if (actual_x < 0)
                pl_shifted_roi_row <= pl_raw_roi_row >> (-actual_x);
            else
                pl_shifted_roi_row <= pl_raw_roi_row;

            pl_rom_data   <= template_ram_rdata;
            pl_valid_mask <= current_x_mask;
        end
    end

    // --- Pipeline Stage 3: Masked Popcount ---
    always_comb begin
        xor_result = (pl_shifted_roi_row ^ pl_rom_data) & pl_valid_mask;
        
        popcount_val = '0;
        for (int i = 0; i < 32; i++) begin
            popcount_val = popcount_val + xor_result[i];
        end
    end

    // --- Valid Signal Shift Register ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pl_eval_valid_1 <= 1'b0;
            pl_eval_valid_2 <= 1'b0;
            eval_valid      <= 1'b0;
        end else if (state == ST_MATCHING) begin
            pl_eval_valid_1 <= (match_row_cnt < 6'd32);
            pl_eval_valid_2 <= pl_eval_valid_1;
            eval_valid      <= pl_eval_valid_2;
        end else begin
            pl_eval_valid_1 <= 1'b0;
            pl_eval_valid_2 <= 1'b0;
            eval_valid      <= 1'b0;
        end
    end

    // --- State Machine & Accumulation ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            template_idx       <= '0;
            match_row_cnt      <= '0;
            dx_idx             <= '0;
            dy_idx             <= '0;
            current_mismatches <= '0;
            min_mismatches     <= 32'hFFFFFFFF;
            min_class_id       <= '0;
            template_ram_addr  <= '0;
        end else begin
            if (state == ST_IDLE) begin
                template_idx       <= '0;
                match_row_cnt      <= '0;
                dx_idx             <= '0;
                dy_idx             <= '0;
                current_mismatches <= '0;
                min_mismatches     <= 32'hFFFFFFFF;
                min_class_id       <= '0;
            end else if (state == ST_MATCHING) begin
                
                if (match_row_cnt < 6'd34) begin
                    if (match_row_cnt < 6'd32) begin
                        template_ram_addr <= (template_idx * 32) + match_row_cnt;
                    end
                    match_row_cnt <= match_row_cnt + 1;
                    
                    if (eval_valid) begin
                        current_mismatches <= current_mismatches + popcount_val;
                    end
                    
                end else begin
                    if (current_mismatches < min_mismatches) begin
                        min_mismatches <= current_mismatches;
                        min_class_id   <= template_idx;
                    end
                    
                    if (dx_idx < 4'd8) begin
                        dx_idx <= dx_idx + 1;
                    end else begin
                        dx_idx <= '0;
                        if (dy_idx < 4'd8) begin
                            dy_idx <= dy_idx + 1;
                        end else begin
                            dy_idx <= '0;
                            template_idx <= template_idx + 1;
                        end
                    end
                    
                    match_row_cnt      <= '0;
                    current_mismatches <= '0;
                end
            end
        end
    end

    assign match_done    = (state == ST_DONE);
    assign best_class_id = min_class_id;
    assign best_score    = min_mismatches;

endmodule