module template_matching_engine #(
    parameter int TARGET_W = 64,
    parameter int TARGET_H = 64,
    parameter int TEMPLATE_COUNT = 50,
    parameter int TEMPLATE_ADDR_W = 10 
)(
    input  logic         clk,
    input  logic         rst_n,

    // Interface from ROI Fetcher
    input  logic [7:0]   s_axis_gray_tdata,
    input  logic         s_axis_gray_tvalid,
    input  logic         s_axis_gray_tlast,
    
    // Interface to Template RAM (Pre-loaded by Host)
    output logic [TEMPLATE_ADDR_W-1:0] template_ram_addr,
    input  logic [31:0]  template_ram_rdata,

    // Outputs
    output logic         match_done,
    output logic [7:0]   best_class_id,
    output logic [31:0]  best_score
);

    // ==========================================
    // FSM States
    // ==========================================
    typedef enum logic [2:0] {
        ST_IDLE          = 3'b000,
        ST_RCV_ROI       = 3'b001,
        ST_CALC_MEAN     = 3'b010,
        ST_BINARIZE      = 3'b011,
        ST_MATCHING      = 3'b100,
        ST_DONE          = 3'b101
    } state_e;

    state_e state, next_state;

    // ==========================================
    // Internal Memories & Registers
    // ==========================================
    logic [11:0] pixel_cnt;
    logic [19:0] roi_sum;
    logic [7:0]  roi_mean;
    
    // ROI Buffer: 64x64 = 4096 pixels
    logic [7:0]  roi_buf [0:4095];
    logic [11:0] buf_rd_addr;
    logic [7:0]  buf_rd_data;
    
    // Binary ROI Shift Registers (Simplified to 32x32 active window for matching)
    logic [31:0] bin_roi [0:31]; 
    logic [4:0]  bin_row_cnt;
    logic [5:0]  bin_col_cnt;

    // Matching Counters
    logic [7:0]  template_idx;
    logic [3:0]  shift_x;
    logic [3:0]  shift_y;
    logic [4:0]  match_row_cnt;
    
    logic [31:0] current_mismatches;
    logic [31:0] min_mismatches;
    logic [7:0]  min_class_id;

    // ==========================================
    // State Machine
    // ==========================================
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
                if (s_axis_gray_tvalid)
                    next_state = ST_RCV_ROI;
            end
            
            ST_RCV_ROI: begin
                if (s_axis_gray_tvalid && s_axis_gray_tlast && pixel_cnt == 4095)
                    next_state = ST_CALC_MEAN;
            end
            
            ST_CALC_MEAN: begin
                next_state = ST_BINARIZE;
            end
            
            ST_BINARIZE: begin
                if (buf_rd_addr == 4095)
                    next_state = ST_MATCHING;
            end
            
            ST_MATCHING: begin
                if (template_idx == TEMPLATE_COUNT)
                    next_state = ST_DONE;
            end
            
            ST_DONE: begin
                next_state = ST_IDLE;
            end
            
            default: next_state = ST_IDLE;
        endcase
    end

    // ==========================================
    // Receive ROI & Accumulate
    // ==========================================
    always_ff @(posedge clk) begin
        if (state == ST_RCV_ROI && s_axis_gray_tvalid) begin
            roi_buf[pixel_cnt] <= s_axis_gray_tdata;
        end
        buf_rd_data <= roi_buf[buf_rd_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_cnt <= '0;
            roi_sum   <= '0;
        end else begin
            if (state == ST_IDLE) begin
                pixel_cnt <= '0;
                roi_sum   <= '0;
            end else if (state == ST_RCV_ROI && s_axis_gray_tvalid) begin
                pixel_cnt <= pixel_cnt + 1;
                roi_sum   <= roi_sum + s_axis_gray_tdata;
            end
        end
    end

    // ==========================================
    // Calculate Mean (Divide by 4096 -> Shift by 12)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            roi_mean <= '0;
        end else if (state == ST_CALC_MEAN) begin
            roi_mean <= roi_sum[19:12];
        end
    end

    // ==========================================
    // Binarization
    // ==========================================
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
                buf_rd_addr <= buf_rd_addr + 1;
                
                // Crop to central 32x32 for the binary mask
                // Original is 64x64, center 32x32 is Y: 16 to 47, X: 16 to 47
                if (buf_rd_addr >= (16*64 + 16) && buf_rd_addr < (48*64 - 16)) begin
                    automatic logic [5:0] x_pos = buf_rd_addr[5:0];
                    if (x_pos >= 16 && x_pos < 48) begin
                        automatic logic bit_val = (buf_rd_data < (roi_mean - 8'd15)) ? 1'b1 : 1'b0;
                        bin_roi[bin_row_cnt][bin_col_cnt] <= bit_val;
                        
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
    end

    // ==========================================
    // Matching Engine (XOR + Popcount)
    // ==========================================
    logic [31:0] xor_result;
    logic [5:0]  popcount_val;

    always_comb begin
        xor_result = bin_roi[match_row_cnt] ^ template_ram_rdata;
        
        // Combinational Popcount (Adder Tree for 32-bits)
        popcount_val = '0;
        for (int i = 0; i < 32; i++) begin
            popcount_val = popcount_val + xor_result[i];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            template_idx       <= '0;
            shift_x            <= '0;
            shift_y            <= '0;
            match_row_cnt      <= '0;
            current_mismatches <= '0;
            min_mismatches     <= 32'hFFFFFFFF;
            min_class_id       <= '0;
            template_ram_addr  <= '0;
        end else begin
            if (state == ST_IDLE) begin
                template_idx       <= '0;
                shift_x            <= '0;
                shift_y            <= '0;
                match_row_cnt      <= '0;
                current_mismatches <= '0;
                min_mismatches     <= 32'hFFFFFFFF;
                min_class_id       <= '0;
            end else if (state == ST_MATCHING) begin
                if (match_row_cnt < 32) begin
                    template_ram_addr  <= (template_idx * 32) + match_row_cnt;
                    current_mismatches <= current_mismatches + popcount_val;
                    match_row_cnt      <= match_row_cnt + 1;
                end else begin
                    // Finished one template comparison
                    if (current_mismatches < min_mismatches) begin
                        min_mismatches <= current_mismatches;
                        min_class_id   <= template_idx;
                    end
                    
                    // Move to next template (Simplified: ignoring spatial shifts for code brevity)
                    template_idx       <= template_idx + 1;
                    match_row_cnt      <= '0;
                    current_mismatches <= '0;
                end
            end
        end
    end

    // ==========================================
    // Outputs
    // ==========================================
    assign match_done    = (state == ST_DONE);
    assign best_class_id = min_class_id;
    assign best_score    = min_mismatches;

endmodule