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

    int f_mask_ref, f_morph_ref, f_ccl_ref;
    int f_mask_actual, f_morph_actual, f_ccl_actual;
    int f_ccl_resolved;  // For resolved labels (Pass 2)
    int f_geom_filtered;  // Geometry filter output
    int mask_errors = 0;
    int morph_errors = 0;
    int ccl_errors = 0;
    int geom_detected_count = 0;  // Count signs passing geometry filter
    
    int f_rtl_roi;
    int f_rtl_results;
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
        f_ccl_ref   = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/ccl_labels_golden.txt", "r");
        
        f_mask_actual  = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/actual_mask_out.txt", "w");
        f_morph_actual = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/actual_morph_out.txt", "w");
        f_ccl_actual   = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/actual_ccl_labels.txt", "w");
        f_ccl_resolved = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/rtl_ccl_resolved.txt", "w");
        
        f_rtl_roi = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/rtl_fetched_gray.hex", "w");
        f_geom_filtered = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/actual_geom_filtered.txt", "w");
        f_rtl_results = $fopen("/users/epnyrk/Project/design/work/ProjectA/data/rtl_final_results.txt", "w");

        if (!f_mask_ref || !f_morph_ref || !f_ccl_ref || !f_rtl_roi || !f_geom_filtered || !f_rtl_results) begin
            $display("ERROR: Could not open reference or data files.");
            $finish;
        end
        
        // Write header to results file
        $fdisplay(f_rtl_results, "===============================================");
        $fdisplay(f_rtl_results, "RTL SIMULATION RESULTS - Road Sign Detector");
        $fdisplay(f_rtl_results, "===============================================");
        $fdisplay(f_rtl_results, "Simulation Time: %0t", $time);
        $fdisplay(f_rtl_results, "Image Resolution: 1920x1080");
        $fdisplay(f_rtl_results, "VERIFICATION GOAL: Detect exactly 1 traffic sign per frame");
        $fdisplay(f_rtl_results, "");
        
        // Write header to geometry filter file
        $fdisplay(f_geom_filtered, "# RTL Geometry Filter Output - Valid Signs Detected");
        $fdisplay(f_geom_filtered, "# Format: label_id area centroid_x centroid_y xmin xmax ymin ymax width height");
        $fdisplay(f_geom_filtered, "# Goal: Detect 1 sign that passes area constraints (50-100k pixels)");
        $fdisplay(f_geom_filtered, "");
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
    int morph_pixel_cnt = 0;

    always @(posedge clk) begin
        if (dut.u_morphology.m_axis_tvalid && dut.u_morphology.m_axis_tready) begin 
            if (dut.u_morphology.m_axis_tuser) morph_sync_done = 1;
            
            if (morph_sync_done) begin
                logic ref_bit;
                int status;
                status = $fscanf(f_morph_ref, "%b", ref_bit);
                morph_pixel_cnt++;
                
                // Write morphology output to file
                $fdisplay(f_morph_actual, "%b", dut.u_morphology.m_axis_tdata);
                
                if (dut.u_morphology.m_axis_tdata !== ref_bit) begin
                    morph_errors++;
                    if (morph_errors < 5)
                        $display("[%0t] MORPH MISMATCH: Expected %b, Got %b", $time, ref_bit, dut.u_morphology.m_axis_tdata);
                end
            end
        end
    end
    
    // Monitor CCL Output (Connected Component Labels)
    int ccl_pixel_cnt = 0;

    always @(posedge clk) begin
        if (dut.u_backend_processing.u_ccl_engine.p1_axis_tvalid) begin
            int ref_label;
            int rtl_label;
            int status;
            status = $fscanf(f_ccl_ref, "%d", ref_label);
            rtl_label = dut.u_backend_processing.u_ccl_engine.p1_axis_tdata;
            ccl_pixel_cnt++;
            
            // Write CCL label to output file
            $fdisplay(f_ccl_actual, "%d", rtl_label);
            
            if (rtl_label !== ref_label) begin
                ccl_errors++;
                if (ccl_errors < 5)
                    $display("[%0t] CCL MISMATCH: Expected %d, Got %d", $time, ref_label, rtl_label);
            end
        end
    end
    
    // Monitor CCL Resolved Output (Pass 2 - Union-Find Results)
    // NOTE: The UF Resolver (ccl_uf_resolver) only modifies the parent RAM (union-find parent pointers).
    //       It does NOT stream resolved pixel labels. To get resolved labels, the CCL Pass 2 must re-stream
    //       Pass 1 labels and apply the resolved parent table for each pixel.
    //       This block monitors the Pass 2 resolved stream if available in your design.
    always @(posedge clk) begin
        if (dut.u_backend_processing.u_ccl_engine.u_resolver.parent_we) begin
            $fdisplay(f_ccl_resolved,"%0d [%0d]", dut.u_backend_processing.u_ccl_engine.u_resolver.parent_addr, dut.u_backend_processing.u_ccl_engine.u_resolver.parent_wdata);
        end
    end

    // Monitor Geometry Filter Output - Components passing area constraints
    // These are the actual detected signs from the RTL filter stage
    // The geometry_filter module outputs: obj_valid, obj_label, obj_xmin, obj_xmax, obj_ymin, obj_ymax
    always @(posedge clk) begin
        if (dut.u_backend_processing.u_geometry_filter.obj_valid) begin
            int label_id    = dut.u_backend_processing.u_geometry_filter.obj_label;
            int bbox_xmin   = dut.u_backend_processing.u_geometry_filter.obj_xmin;
            int bbox_xmax   = dut.u_backend_processing.u_geometry_filter.obj_xmax;
            int bbox_ymin   = dut.u_backend_processing.u_geometry_filter.obj_ymin;
            int bbox_ymax   = dut.u_backend_processing.u_geometry_filter.obj_ymax;
            
            int width       = bbox_xmax - bbox_xmin + 1;
            int height      = bbox_ymax - bbox_ymin + 1;
            int centroid_x  = (bbox_xmin + bbox_xmax) / 2;  // Computed from bbox
            int centroid_y  = (bbox_ymin + bbox_ymax) / 2;  // Computed from bbox
            int area        = width * height;               // Spatial bounding box area
            
            $fdisplay(f_geom_filtered, "%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d", 
                      label_id, area, centroid_x, centroid_y, 
                      bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax, width, height);
            
            geom_detected_count++;
            
            $display("[%0t] GEOMETRY FILTER OUTPUT: Label=%0d, Area=%0d, BBox=[%0d:%0d, %0d:%0d]", 
                     $time, label_id, area, bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax);
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
    logic [7:0] class_id;
    logic [15:0] bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax;

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
        class_id = (read_val >> 8) & 8'hFF;
        $display("Class ID    : %0d (%s)", class_id, template_desc[class_id]); 
        $fdisplay(f_rtl_results, "CLASS_ID: %0d", class_id);
        
        axi_lite_read(12'h008, read_val); 
        bbox_xmin = read_val & 16'hFFFF;
        bbox_xmax = (read_val >> 16) & 16'hFFFF;
        $display("BBox X-Axis : [%0d, %0d]", bbox_xmin, bbox_xmax);
        $fdisplay(f_rtl_results, "BBOX_X: [%0d, %0d]", bbox_xmin, bbox_xmax);
        
        axi_lite_read(12'h00C, read_val); 
        bbox_ymin = read_val & 16'hFFFF;
        bbox_ymax = (read_val >> 16) & 16'hFFFF;
        $display("BBox Y-Axis : [%0d, %0d]", bbox_ymin, bbox_ymax);
        $fdisplay(f_rtl_results, "BBOX_Y: [%0d, %0d]", bbox_ymin, bbox_ymax);
        $display("----------------------------------------");

        $display("========================================");
        $display("        INTERMEDIATE STAGE REPORT       ");
        $display("========================================");
        if (mask_errors == 0) $display("RED MASK:   MATCHED! (Passed)");
        else                  $display("RED MASK:   FAILED with %0d mismatches", mask_errors);
        
        if (morph_errors == 0) $display("MORPHOLOGY: MATCHED! (Passed)");
        else                   $display("MORPHOLOGY: FAILED with %0d mismatches", morph_errors);
        
        if (ccl_errors == 0) $display("CCL LABELS: MATCHED! (Passed)");
        else                 $display("CCL LABELS: FAILED with %0d mismatches", ccl_errors);
        
        $display("========================================");

        // Write stage results to rtl_results file
        $fdisplay(f_rtl_results, "");
        $fdisplay(f_rtl_results, "INTERMEDIATE_STAGES:");
        $fdisplay(f_rtl_results, "MASK_ERRORS: %0d", mask_errors);
        $fdisplay(f_rtl_results, "MORPH_ERRORS: %0d", morph_errors);
        $fdisplay(f_rtl_results, "CCL_ERRORS: %0d", ccl_errors);
        $fdisplay(f_rtl_results, "MORPH_PIXELS_PROCESSED: %0d", morph_pixel_cnt);
        $fdisplay(f_rtl_results, "CCL_PIXELS_PROCESSED: %0d", ccl_pixel_cnt);
        $fdisplay(f_rtl_results, "ROI_PIXELS_FETCHED: %0d", roi_pixel_cnt);
        $fdisplay(f_rtl_results, "");
        $fdisplay(f_rtl_results, "GEOMETRY_FILTER_RESULTS:");
        $fdisplay(f_rtl_results, "SIGNS_DETECTED: %0d (Goal: exactly 1 sign)", geom_detected_count);
        if (geom_detected_count == 1)
            $fdisplay(f_rtl_results, "GEOMETRY_FILTER: PASS (detected 1 sign)");
        else if (geom_detected_count == 0)
            $fdisplay(f_rtl_results, "GEOMETRY_FILTER: FAIL (detected 0 signs - NO DETECTION)");
        else
            $fdisplay(f_rtl_results, "GEOMETRY_FILTER: FAIL (detected %0d signs - should be 1)", geom_detected_count);
        $fdisplay(f_rtl_results, "");
        $fdisplay(f_rtl_results, "FINAL_DETECTION:");
        $fdisplay(f_rtl_results, "CLASS_NAME: %s", template_desc[class_id]);
        $fdisplay(f_rtl_results, "BBOX_DIMENSIONS:");
        $fdisplay(f_rtl_results, "  X: [%0d, %0d] (width=%0d)", bbox_xmin, bbox_xmax, (bbox_xmax - bbox_xmin + 1));
        $fdisplay(f_rtl_results, "  Y: [%0d, %0d] (height=%0d)", bbox_ymin, bbox_ymax, (bbox_ymax - bbox_ymin + 1));
        $fdisplay(f_rtl_results, "");
        $fdisplay(f_rtl_results, "VERIFICATION_STATUS:");
        if (geom_detected_count == 1 && mask_errors == 0 && morph_errors == 0 && ccl_errors == 0)
            $fdisplay(f_rtl_results, "  Overall: PASS (1 sign detected and identified)");
        else if (geom_detected_count == 1)
            $fdisplay(f_rtl_results, "  Overall: PARTIAL (1 sign detected, but intermediate stages have errors)");
        else if (geom_detected_count == 0)
            $fdisplay(f_rtl_results, "  Overall: FAIL (No signs detected at geometry filter)");
        else
            $fdisplay(f_rtl_results, "  Overall: FAIL (%0d signs detected, should be 1)", geom_detected_count);
        $fdisplay(f_rtl_results, "===============================================");
        
        $fclose(f_mask_ref);
        $fclose(f_morph_ref);
        $fclose(f_ccl_ref);
        $fclose(f_mask_actual);
        $fclose(f_morph_actual);
        $fclose(f_ccl_actual);
        $fclose(f_ccl_resolved);
        $fclose(f_rtl_roi);
        $fclose(f_geom_filtered);
        $fclose(f_rtl_results);
        
        $display("");
        $display("========== SIMULATION OUTPUT FILES SUMMARY ==========");
        $display("Dumped %0d ROI pixels to rtl_fetched_gray.hex", roi_pixel_cnt);
        $display("Dumped %0d CCL Pass 1 labels to actual_ccl_labels.txt", ccl_pixel_cnt);
        $display("Dumped CCL Pass 2 (resolved) labels to rtl_ccl_resolved.txt");
        $display("Dumped %0d detected signs to actual_geom_filtered.txt", geom_detected_count);
        $display("RTL final results saved to rtl_final_results.txt");
        $display("======================================================");
        
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