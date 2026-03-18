`timescale 1ns / 1ps

module tb_backend_roi_matcher;

    // --- Parameters ---
    localparam int AXI_ADDR_W = 32;
    localparam int AXI_DATA_W = 64; 
    localparam int FRAME_W    = 2872;
    localparam int FRAME_H    = 1617;
    localparam int TOTAL_PIXELS = FRAME_W * FRAME_H;
    localparam int ROM_DEPTH  = 1600;

    // --- Signals ---
    logic clk, rst_n;
    logic start_trigger;
    logic [31:0] base_addr;
    logic [15:0] roi_xmin, roi_xmax, roi_ymin, roi_ymax;

    // AXI Interface
    logic [AXI_ADDR_W-1:0] m_axi_araddr;
    logic [AXI_DATA_W-1:0] m_axi_rdata;
    logic m_axi_arvalid, m_axi_arready, m_axi_rvalid, m_axi_rready;

    // Internal Streaming (Gray)
    logic [7:0] gray_data;
    logic       gray_valid, gray_last;

    // Internal ROM Interface
    logic [10:0] tmpl_addr;
    logic [31:0] tmpl_rdata;

    // Outputs
    logic fetch_done, match_done;
    logic [7:0]  best_class_id;
    logic [31:0] best_score;

    // Memory Models
    logic [63:0] main_ddr [0:(TOTAL_PIXELS/2)-1];
    logic [31:0] template_rom [0:ROM_DEPTH-1];

    // Clock Generation
    always #5 clk = (clk === 1'b0);

    // File handles
    int fd_gray, fd_bin, fd_res;

    // ==========================================
    // 1. Memory Initialization
    // ==========================================
    initial begin
        logic [23:0] raw_pixel_mem [0:TOTAL_PIXELS-1];
        
        // Load Templates
        $display("[%0t] Loading templates.mem...", $time);
        $readmemh("design/work/ProjectA/data/templates.mem", template_rom);
        
        // Load Image and Pack into 64-bit DDR
        $display("[%0t] Loading image_in.hex...", $time);
        $readmemh("design/work/ProjectA/data/image_in.hex", raw_pixel_mem);
        for (int i = 0; i < TOTAL_PIXELS/2; i++) begin
            main_ddr[i] = {8'h00, raw_pixel_mem[i*2+1], 8'h00, raw_pixel_mem[i*2]};
        end
        $display("[%0t] Memory Ready.", $time);
    end

    // ROM Read Logic
    always_ff @(posedge clk) tmpl_rdata <= template_rom[tmpl_addr];

    // ==========================================
    // 2. AXI Slave Mock (DDR Controller)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 0;
            m_axi_rvalid  <= 0;
            m_axi_rdata   <= 0;
        end else begin
            // Address Handshake
            if (m_axi_arvalid && !m_axi_arready) m_axi_arready <= 1'b1;
            else m_axi_arready <= 1'b0;

            // Data Handshake
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rvalid <= 1'b1;
                // Select Lower/Upper 32-bit pixel from 64-bit word
                m_axi_rdata <= m_axi_araddr[2] ? {32'h0, main_ddr[m_axi_araddr >> 3][63:32]} : 
                                               {32'h0, main_ddr[m_axi_araddr >> 3][31:0]};
            end 
            else if (m_axi_rready && m_axi_rvalid) begin
                m_axi_rvalid <= 1'b0;
            end
        end
    end

    // ==========================================
    // 3. DUT Instantiations
    // ==========================================
    roi_fetcher_axi_master #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .FRAME_W(FRAME_W)
    ) u_fetcher (
        .clk(clk), .rst_n(rst_n),
        .start(start_trigger), .base_addr(base_addr),
        .roi_xmin(roi_xmin), .roi_xmax(roi_xmax),
        .roi_ymin(roi_ymin), .roi_ymax(roi_ymax),
        .m_axi_araddr(m_axi_araddr), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axis_gray_tdata(gray_data), .m_axis_gray_tvalid(gray_valid), .m_axis_gray_tlast(gray_last),
        .fetch_done(fetch_done),
        .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(), .m_axi_rresp(), .m_axi_rlast()
    );

    template_matching_engine #(
        .TEMPLATE_COUNT(19), .TEMPLATE_ADDR_W(11)
    ) u_matcher (
        .clk(clk), .rst_n(rst_n),
        .s_axis_gray_tdata(gray_data), .s_axis_gray_tvalid(gray_valid), .s_axis_gray_tlast(gray_last),
        .template_ram_addr(tmpl_addr), .template_ram_rdata(tmpl_rdata),
        .match_done(match_done), .best_class_id(best_class_id), .best_score(best_score)
    );

    // ==========================================
    // 4. File Logging (WITH FORCED FLUSH)
    // ==========================================
    initial begin
        fd_gray = $fopen("design/work/ProjectA/data/rtl_fetched_gray.hex", "w");
        fd_bin  = $fopen("design/work/ProjectA/data/rtl_binary_roi.hex", "w");
        fd_res  = $fopen("design/work/ProjectA/data/rtl_final_results.txt", "w");
        if (fd_gray == 0) $display("ERROR: Could not create output files.");
    end

    // Log Gray pixels with immediate flush to disk
    always @(posedge clk) begin
        if (gray_valid && gray_data !== 8'hxx) begin
            $fdisplay(fd_gray, "%02X", gray_data);
            $fflush(fd_gray); 
        end
    end

    // Monitor Matcher internal progress
    always @(posedge clk) begin
        if (u_matcher.state != u_matcher.next_state) begin
            $display("[%0t] MATCHER_TRACE: State %0d -> %0d", $time, u_matcher.state, u_matcher.next_state);
        end
        
        if (u_matcher.state == 1) begin // ST_RCV_ROI
            if (gray_valid) 
                $display("[%0t] MATCHER_PIXELS: Got pixel %0d/4096, tlast=%b", $time, u_matcher.pixel_cnt, gray_last);
        end
    end

    // Improved Binary Logging
    logic bin_written = 0; // Flag to write only once

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bin_written <= 0;
        end else begin
            // Trigger write exactly when entering state 4
            if (u_matcher.state == 4 && !bin_written) begin
                $display("[%0t] TB_INFO: Capturing Binary Mask...", $time);
                for (int r = 0; r < 32; r++) begin
                    $fdisplay(fd_bin, "%08X", u_matcher.bin_roi[r]);
                end
                $fflush(fd_bin);
                bin_written <= 1; // Prevent multiple writes
            end
        end
    end

    // Log Binary Matrix when matching starts
    always @(posedge clk) begin
        if (u_matcher.state == 4 && u_matcher.match_row_cnt == 0 && u_matcher.template_idx == 0) begin
            for (int r = 0; r < 32; r++) begin
                $fdisplay(fd_bin, "%08X", u_matcher.bin_roi[r]);
            end
            $fflush(fd_bin);
        end
    end

    // ==========================================
    // Monitor: Print Best Score per Template
    // ==========================================
    integer current_tmpl_min = 32'hFFFFFFFF;

    always @(posedge clk) begin
        // Check if we are in ST_MATCHING (3'b100) and at the evaluation cycle (34)
        if (u_matcher.state == 3'b100 && u_matcher.match_row_cnt == 6'd34) begin
            
            // Determine the minimum score for the current template so far
            automatic integer temp_min = (u_matcher.current_mismatches < current_tmpl_min) ? 
                                          u_matcher.current_mismatches : current_tmpl_min;
            
            // If this is the last shift (dx=8, dy=8) for the current template
            if (u_matcher.dx_idx == 4'd8 && u_matcher.dy_idx == 4'd8) begin
                $display("[%0t] RTL_SCORE_TABLE: Template %2d | Mismatches: %0d", 
                         $time, u_matcher.template_idx, temp_min);
                
                // Reset the local minimum for the next template
                current_tmpl_min = 32'hFFFFFFFF;
            end else begin
                // Update local minimum and continue to next shift
                current_tmpl_min = temp_min;
            end
        end
        
        // Reset outside of matching state
        if (u_matcher.state == 3'b000) begin
            current_tmpl_min = 32'hFFFFFFFF;
        end
    end

    // ==========================================
    // 5. Test Stimulus
    // ==========================================
    initial begin
        int fd_in, status, d_lbl, d_w, d_h, d_area, d_ar;
        clk = 0; rst_n = 0; start_trigger = 0; base_addr = 0;

        // Auto-extract ROI coordinates
        fd_in = $fopen("design/work/ProjectA/data/detected_boxes.txt", "r");
        if (fd_in) begin
            status = $fscanf(fd_in, "%d,%d,%d,%d,%d,%d,%d,%d,%d", d_lbl, roi_xmin, roi_ymin, roi_xmax, roi_ymax, d_w, d_h, d_area, d_ar);
            $fclose(fd_in);
            $display("[%0t] ROI LOADED: X[%0d:%0d] Y[%0d:%0d]", $time, roi_xmin, roi_xmax, roi_ymin, roi_ymax);
        end

        // Reset Sequence
        #100 rst_n = 1;
        #500;

        // Start Operation
        $display("[%0t] Triggering Start...", $time);
        @(posedge clk); start_trigger = 1;
        repeat(5) @(posedge clk); start_trigger = 0;

        // Wait for Fetcher
        wait(fetch_done);
        $display("[%0t] Fetcher Finished Success.", $time);

        // Wait for Matcher with safety timeout
        fork
            begin
                wait(match_done);
                $display("[%0t] Matcher Finished Success. Best ID: %0d", $time, best_class_id);
                $fdisplay(fd_res, "Best ID: %0d\nScore: %0d", best_class_id, best_score);
                $fflush(fd_res);
            end
            begin
                #10ms;
                $display("[%0t] WARNING: Matcher timed out or hanging.", $time);
            end
        join_any

        // Close files properly
        #1000;
        $fclose(fd_gray);
        $fclose(fd_bin);
        $fclose(fd_res);
        $display("[%0t] Simulation Finished. Files closed and flushed.", $time);
        $finish;
    end

endmodule