`timescale 1ns / 1ps

module tb_road_sign_detector_top();

    // ==========================================
    // Parameters
    // ==========================================
    parameter int AXI_LITE_ADDR_W = 12;
    parameter int AXI_LITE_DATA_W = 32;
    parameter int AXIS_TDATA_W    = 24;
    parameter int AXI_FULL_ADDR_W = 32;
    parameter int AXI_FULL_DATA_W = 64;
    parameter int IMG_W           = 1920;
    parameter int IMG_H           = 1080;
    parameter int CLK_PERIOD      = 10;

    int f_mask_ref, f_morph_ref;
    int f_mask_actual, f_morph_actual;
    int mask_errors = 0;
    int morph_errors = 0;
    
    int f_rtl_roi;
    int roi_pixel_cnt = 0;

    // Array for fast image streaming
    logic [31:0] image_array [0:IMG_W*IMG_H-1];
    
    // Array for Template Descriptions
    string template_desc [0:19];

    // ==========================================
    // Signals
    // ==========================================
    logic clk = 0;
    logic rst_n = 0;

    // AXI4-Lite Slave Interface (Control & Status)
    logic [AXI_LITE_ADDR_W-1:0] s_axi_awaddr = 0;
    logic                       s_axi_awvalid = 0;
    logic                       s_axi_awready;
    logic [AXI_LITE_DATA_W-1:0] s_axi_wdata = 0;
    logic [3:0]                 s_axi_wstrb = 0;
    logic                       s_axi_wvalid = 0;
    logic                       s_axi_wready;
    logic [1:0]                 s_axi_bresp;
    logic                       s_axi_bvalid;
    logic                       s_axi_bready = 0;
    
    logic [AXI_LITE_ADDR_W-1:0] s_axi_araddr = 0;
    logic                       s_axi_arvalid = 0;
    logic                       s_axi_arready;
    logic [AXI_LITE_DATA_W-1:0] s_axi_rdata;
    logic [1:0]                 s_axi_rresp;
    logic                       s_axi_rvalid;
    logic                       s_axi_rready = 0;

    // AXI4-Stream Slave Interface (Raw RGB)
    logic [AXIS_TDATA_W-1:0]    s_axis_tdata = 0;
    logic                       s_axis_tvalid = 0;
    logic                       s_axis_tready;
    logic                       s_axis_tuser = 0;
    logic                       s_axis_tlast = 0;

    // AXI4 Master 0: RGB Frame Writer
    logic [AXI_FULL_ADDR_W-1:0] m0_axi_awaddr;
    logic [7:0]                 m0_axi_awlen;
    logic [2:0]                 m0_axi_awsize;
    logic [1:0]                 m0_axi_awburst;
    logic                       m0_axi_awvalid;
    logic                       m0_axi_awready;
    logic [AXI_FULL_DATA_W-1:0] m0_axi_wdata;
    logic [7:0]                 m0_axi_wstrb;
    logic                       m0_axi_wlast;
    logic                       m0_axi_wvalid;
    logic                       m0_axi_wready;
    logic [1:0]                 m0_axi_bresp;
    logic                       m0_axi_bvalid;
    logic                       m0_axi_bready;

    // AXI4 Master 1: Backend Processing
    logic [AXI_FULL_ADDR_W-1:0] m1_axi_awaddr;
    logic [7:0]                 m1_axi_awlen;
    logic [2:0]                 m1_axi_awsize;
    logic [1:0]                 m1_axi_awburst;
    logic                       m1_axi_awvalid;
    logic                       m1_axi_awready;
    logic [AXI_FULL_DATA_W-1:0] m1_axi_wdata;
    logic [7:0]                 m1_axi_wstrb;
    logic                       m1_axi_wlast;
    logic                       m1_axi_wvalid;
    logic                       m1_axi_wready;
    logic [1:0]                 m1_axi_bresp;
    logic                       m1_axi_bvalid;
    logic                       m1_axi_bready;
    
    logic [AXI_FULL_ADDR_W-1:0] m1_axi_araddr;
    logic [7:0]                 m1_axi_arlen;
    logic [2:0]                 m1_axi_arsize;
    logic [1:0]                 m1_axi_arburst;
    logic                       m1_axi_arvalid;
    logic                       m1_axi_arready;
    logic [AXI_FULL_DATA_W-1:0] m1_axi_rdata;
    logic                       m1_axi_rlast;
    logic [1:0]                 m1_axi_rresp;
    logic                       m1_axi_rvalid;
    logic                       m1_axi_rready;

    // Interrupt
    logic irq;

    // Unified Memory Model
    logic [63:0] ddr_memory [logic [31:0]];

    // Clock Generator
    always #(CLK_PERIOD/2) clk = ~clk;

    // ==========================================
    // DUT Instantiation
    // ==========================================
    road_sign_detector_top #(
        .AXI_LITE_ADDR_W(AXI_LITE_ADDR_W),
        .AXI_LITE_DATA_W(AXI_LITE_DATA_W),
        .AXIS_TDATA_W(AXIS_TDATA_W),
        .AXI_FULL_ADDR_W(AXI_FULL_ADDR_W),
        .AXI_FULL_DATA_W(AXI_FULL_DATA_W),
        .IMG_W(IMG_W),
        .IMG_H(IMG_H)
    ) dut (.*);

    logic [31:0] final_score_debug;
    logic [7:0]  final_index_debug;

    always @(posedge clk) begin
        if (dut.u_backend_processing.u_matcher.state == 3'b101) begin // ST_DONE
            final_score_debug <= dut.u_backend_processing.u_matcher.best_score;
            final_index_debug <= dut.u_backend_processing.u_matcher.template_idx;
        end
    end

   // ==========================================
    // AXI Slave Memory Model - M0 (Write Only)
    // ==========================================
    logic [AXI_FULL_ADDR_W-1:0] m0_awaddr_latch;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m0_axi_awready <= 1'b1;
            m0_axi_wready  <= 1'b1;
            m0_axi_bvalid  <= 1'b0;
        end else begin
            if (m0_axi_awvalid && m0_axi_awready) begin
                m0_awaddr_latch <= m0_axi_awaddr;
            end
            
            if (m0_axi_wvalid && m0_axi_wready) begin
                logic [AXI_FULL_ADDR_W-1:0] active_addr;
                active_addr = (m0_axi_awvalid) ? m0_axi_awaddr : m0_awaddr_latch;
                
                ddr_memory[active_addr & ~(32'h7)] <= m0_axi_wdata;
            end
            
            if (m0_axi_wvalid && m0_axi_wready && m0_axi_wlast) begin
                m0_axi_bvalid <= 1'b1;
            end else if (m0_axi_bready && m0_axi_bvalid) begin
                m0_axi_bvalid <= 1'b0;
            end
        end
    end

    // ==========================================
    // AXI Slave Memory Model - M1 (Read/Write)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1_axi_awready <= 1'b0;
            m1_axi_wready  <= 1'b0;
            m1_axi_bvalid  <= 1'b0;
            m1_axi_bresp   <= 2'b00;
        end else begin
            m1_axi_awready <= 1'b1;
            m1_axi_wready  <= 1'b1;
            
            if (m1_axi_wvalid && m1_axi_wready && m1_axi_awvalid && m1_axi_awready) begin
                ddr_memory[m1_axi_awaddr & ~(32'h7)] <= m1_axi_wdata;
            end
            
            if (m1_axi_wvalid && m1_axi_wready && m1_axi_wlast) begin
                m1_axi_bvalid <= 1'b1;
            end else if (m1_axi_bready && m1_axi_bvalid) begin
                m1_axi_bvalid <= 1'b0;
            end
        end
    end

    logic [AXI_FULL_ADDR_W-1:0] m1_read_addr_reg;
    logic m1_pending_read;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1_axi_arready <= 1'b0;
            m1_axi_rvalid  <= 1'b0;
            m1_axi_rlast   <= 1'b0;
            m1_axi_rresp   <= 2'b00;
            m1_axi_rdata   <= '0;
            m1_pending_read<= 1'b0;
        end else begin
            m1_axi_arready <= !m1_pending_read;
            
            if (m1_axi_arvalid && m1_axi_arready) begin
                m1_pending_read  <= 1'b1;
                m1_read_addr_reg <= m1_axi_araddr & ~(32'h7);
            end
            
            if (m1_pending_read) begin
                m1_axi_rvalid <= 1'b1;
                m1_axi_rlast  <= 1'b1;
                
                if (ddr_memory.exists(m1_read_addr_reg)) begin
                    m1_axi_rdata <= ddr_memory[m1_read_addr_reg];
                end else begin
                    m1_axi_rdata <= 64'h0; 
                end
                
                if (m1_axi_rvalid && m1_axi_rready) begin
                    m1_axi_rvalid   <= 1'b0;
                    m1_axi_rlast    <= 1'b0;
                    m1_pending_read <= 1'b0;
                end
            end
        end
    end

    // ==========================================
    // AXI-Lite Master Tasks
    // ==========================================
    task axi_lite_write(input [11:0] addr, input [31:0] data);
        @(posedge clk);
        s_axi_awaddr  <= addr;
        s_axi_awvalid <= 1'b1;
        s_axi_wdata   <= data;
        s_axi_wstrb   <= 4'hF;
        s_axi_wvalid  <= 1'b1;
        
        wait(s_axi_awready && s_axi_wready);
        @(posedge clk);
        s_axi_awvalid <= 1'b0;
        s_axi_wvalid  <= 1'b0;
        s_axi_bready  <= 1'b1;
        
        wait(s_axi_bvalid);
        @(posedge clk);
        s_axi_bready  <= 1'b0;
    endtask

    task axi_lite_read(input [11:0] addr, output [31:0] data);
        @(posedge clk);
        s_axi_araddr  <= addr;
        s_axi_arvalid <= 1'b1;
        
        wait(s_axi_arready);
        @(posedge clk);
        s_axi_arvalid <= 1'b0;
        s_axi_rready  <= 1'b1;
        
        wait(s_axi_rvalid);
        data = s_axi_rdata;
        @(posedge clk);
        s_axi_rready  <= 1'b0;
    endtask

    // ==========================================
    // Intermediate Signals Monitoring & Comparison
    // ==========================================
    initial begin
        int fd_map;
        string line_str;
        int id_val;
        int dash_idx;
        string desc_str;

        wait(rst_n == 1'b1);
        
        for (int i = 0; i < 20; i++) template_desc[i] = "Unknown";

        fd_map = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/template_mapping.txt", "r");
        if (fd_map) begin
            while (!$feof(fd_map)) begin
                void'($fgets(line_str, fd_map));
                if ($sscanf(line_str, "ID %d :", id_val) == 1) begin
                    dash_idx = 0;
                    for (int i = 0; i < line_str.len(); i++) begin
                        if (line_str[i] == "-") begin
                            dash_idx = i;
                            break;
                        end
                    end
                    if (dash_idx > 0 && id_val < 20) begin
                        desc_str = line_str.substr(dash_idx + 1, line_str.len() - 1);
                        
                        while(desc_str.len() > 0 && (desc_str[0] == " " || desc_str[0] == "\t"))
                            desc_str = desc_str.substr(1, desc_str.len() - 1);
                            
                        while(desc_str.len() > 0 && (desc_str[desc_str.len()-1] == "\n" || desc_str[desc_str.len()-1] == "\r" || desc_str[desc_str.len()-1] == " "))
                            desc_str = desc_str.substr(0, desc_str.len() - 2);

                        template_desc[id_val] = desc_str;
                    end
                end
            end
            $fclose(fd_map);
        end else begin
            $display("WARNING: Could not open template_mapping.txt");
        end

        f_mask_ref  = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/mask_out2.txt", "r");
        f_morph_ref = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/morph_out2.txt", "r");
        
        f_mask_actual  = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/actual_mask_out.txt", "w");
        f_morph_actual = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/actual_morph_out.txt", "w");
        
        f_rtl_roi = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/rtl_fetched_gray.hex", "w");

        if (!f_mask_ref || !f_morph_ref || !f_rtl_roi) begin
            $display("ERROR: Could not open reference or data files.");
            $finish;
        end
    end

    // Monitor Red Mask Output
    always @(posedge clk) begin
        if (dut.u_red_mask.m_axis_tvalid && dut.u_morphology.s_axis_tready) begin 
            logic ref_bit;
            int status;
            status = $fscanf(f_mask_ref, "%b", ref_bit);
            $fdisplay(f_mask_actual, "%b", dut.u_red_mask.m_axis_tdata);
            if (dut.u_red_mask.m_axis_tdata !== ref_bit) begin
                mask_errors++;
            end
        end
    end

    // Monitor Morphology Output
    bit morph_sync_done = 0;

    always @(posedge clk) begin
        if (dut.u_morphology.m_axis_tvalid && dut.u_morphology.m_axis_tready) begin 
            if (dut.u_morphology.m_axis_tuser) morph_sync_done = 1;
            
            if (morph_sync_done) begin
                logic ref_bit;
                int status;
                status = $fscanf(f_morph_ref, "%b", ref_bit);
                
                if (dut.u_morphology.m_axis_tdata !== ref_bit) begin
                    morph_errors++;
                    if (morph_errors < 5)
                        $display("[%0t] MORPH MISMATCH: Expected %b, Got %b", $time, ref_bit, dut.u_morphology.m_axis_tdata);
                end
            end
        end
    end
    
    // ==========================================
    // Monitor: Dump fetched ROI to Hex File
    // ==========================================
    always @(posedge clk) begin
        if (dut.u_backend_processing.u_matcher.state == 3'b001) begin 
            if (dut.u_backend_processing.u_matcher.s_axis_gray_tvalid) begin 
                $fdisplay(f_rtl_roi, "%02x", dut.u_backend_processing.u_matcher.s_axis_gray_tdata);
                roi_pixel_cnt++;
            end
        end
    end

    // ==========================================
    // Main Test Sequence
    // ==========================================
    logic [31:0] read_val;

    initial begin
        $display("[%0t] System Asserting Reset...", $time);
        rst_n = 0;
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 10);
        
        $display("[%0t] Configuring Base Address to 0x8000_0000", $time);
        axi_lite_write(12'h014, 32'h8000_0000); 
        
        $display("[%0t] Enabling System", $time);
        axi_lite_write(12'h000, 32'h0000_0001); 

        $display("[%0t] Starting Video RGB Stream...", $time);
        stream_rgb_file("/users/epnyrk/Project/design/work/ProjectA/data/image_in2.hex");
        $display("[%0t] Video Stream Complete. Waiting for Processing...", $time);
        
        wait(irq == 1'b1);
        $display("[%0t] IRQ Received! Processing complete.", $time);
        
        $display("----------------------------------------");
        $display("DIRECT PROBE FROM HARDWARE");
        $display("Class ID    : %0d (%s)", dut.sts_best_class_id, template_desc[dut.sts_best_class_id]);
        $display("BBox X-Axis : [%0d, %0d]", dut.sts_bbox_xmin, dut.sts_bbox_xmax);
        $display("BBox Y-Axis : [%0d, %0d]", dut.sts_bbox_ymin, dut.sts_bbox_ymax);
        $display("----------------------------------------");

        axi_lite_write(12'h000, 32'h0000_0003); 

        axi_lite_read(12'h004, read_val); 
        $display("----------------------------------------");
        $display("Class ID    : %0d (%s)", (read_val >> 8) & 8'hFF, template_desc[(read_val >> 8) & 8'hFF]); 
        
        axi_lite_read(12'h008, read_val); 
        $display("BBox X-Axis : [%0d, %0d]", read_val & 16'hFFFF, (read_val >> 16) & 16'hFFFF);
        
        axi_lite_read(12'h00C, read_val); 
        $display("BBox Y-Axis : [%0d, %0d]", read_val & 16'hFFFF, (read_val >> 16) & 16'hFFFF);
        $display("----------------------------------------");

        $display("========================================");
        $display("        INTERMEDIATE STAGE REPORT       ");
        $display("========================================");
        if (mask_errors == 0) $display("RED MASK:   MATCHED! (Passed)");
        else                  $display("RED MASK:   FAILED with %0d mismatches", mask_errors);
        
        if (morph_errors == 0) $display("MORPHOLOGY: MATCHED! (Passed)");
        else                   $display("MORPHOLOGY: FAILED with %0d mismatches", morph_errors);
        $display("========================================");

        $fclose(f_mask_ref);
        $fclose(f_morph_ref);
        $fclose(f_mask_actual);
        $fclose(f_morph_actual);
        
        $fclose(f_rtl_roi);
        $display("Dumped %0d ROI pixels to rtl_fetched_gray.hex", roi_pixel_cnt);
        
        #(CLK_PERIOD * 100);
        $finish;
    end

    // ==========================================
    // Monitor: Print Best Score per Template
    // ==========================================
    integer current_tmpl_min = 32'hFFFFFFFF;

    always @(posedge clk) begin
        if (dut.u_backend_processing.u_matcher.state == 3'b100 && dut.u_backend_processing.u_matcher.match_row_cnt == 6'd34) begin
            
            automatic integer temp_min = (dut.u_backend_processing.u_matcher.current_mismatches < current_tmpl_min) ? 
                                          dut.u_backend_processing.u_matcher.current_mismatches : current_tmpl_min;
            
            if (dut.u_backend_processing.u_matcher.dx_idx == 4'd8 && dut.u_backend_processing.u_matcher.dy_idx == 4'd8) begin
                $display("[%0t] RTL_SCORE_TABLE: Template %2d (%s) | Mismatches: %0d", 
                         $time, dut.u_backend_processing.u_matcher.template_idx, 
                         template_desc[dut.u_backend_processing.u_matcher.template_idx], temp_min);
                
                current_tmpl_min = 32'hFFFFFFFF;
            end else begin
                current_tmpl_min = temp_min;
            end
        end

        if (dut.u_backend_processing.u_matcher.state == 3'b000) begin
            current_tmpl_min = 32'hFFFFFFFF;
        end
    end

    // ==========================================
    // Optimized Stream Task (Loads array into memory first)
    // ==========================================
    task stream_rgb_file(string filename);
        int idx = 0;
        
        $display("[%0t] Loading image into memory array... This is much faster!", $time);
        $readmemh(filename, image_array);
        $display("[%0t] Loading complete. Streaming to DUT...", $time);

        for (int y = 0; y < IMG_H; y++) begin
            for (int x = 0; x < IMG_W; x++) begin
                s_axis_tvalid <= 1'b1;
                s_axis_tdata  <= image_array[idx][23:0]; // Fetch from memory array
                s_axis_tuser  <= (x == 0 && y == 0) ? 1'b1 : 1'b0;
                s_axis_tlast  <= (x == IMG_W - 1) ? 1'b1 : 1'b0;
                
                do begin
                    @(posedge clk);
                end while (!s_axis_tready);
                
                idx++;
            end
        end
        
        s_axis_tvalid <= 1'b0;
        s_axis_tuser  <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

endmodule