module road_sign_detector_top_tb();

    localparam int AXI_LITE_ADDR_W = 12;
    localparam int AXI_LITE_DATA_W = 32;
    localparam int AXIS_TDATA_W    = 24;
    localparam int AXI_FULL_ADDR_W = 32;
    localparam int AXI_FULL_DATA_W = 64;
    localparam int IMG_W           = 2872;
    localparam int IMG_H           = 1617;
    localparam int CLK_PERIOD      = 10;
    localparam int MEM_SIZE        = 64 * 1024 * 1024; 

    logic clk;
    logic rst_n;
    
    logic [AXI_LITE_ADDR_W-1:0] s_axi_awaddr;
    logic                       s_axi_awvalid;
    logic                       s_axi_awready;
    logic [AXI_LITE_DATA_W-1:0] s_axi_wdata;
    logic [3:0]                 s_axi_wstrb;
    logic                       s_axi_wvalid;
    logic                       s_axi_wready;
    logic [1:0]                 s_axi_bresp;
    logic                       s_axi_bvalid;
    logic                       s_axi_bready;
    
    logic [AXI_LITE_ADDR_W-1:0] s_axi_araddr;
    logic                       s_axi_arvalid;
    logic                       s_axi_arready;
    logic [AXI_LITE_DATA_W-1:0] s_axi_rdata;
    logic [1:0]                 s_axi_rresp;
    logic                       s_axi_rvalid;
    logic                       s_axi_rready;
    
    logic [AXIS_TDATA_W-1:0]    s_axis_tdata;
    logic                       s_axis_tvalid;
    logic                       s_axis_tready;
    logic                       s_axis_tuser;
    logic                       s_axis_tlast;
    
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
    
    logic                       irq;

    int fd_in;
    int fd_mask_out;
    int fd_morph_out;
    int fd_geom_out;
    int fd_match_out;
    int fd_labels_out;

    logic [7:0] ram [0:MEM_SIZE-1];

    road_sign_detector_top #(
        .AXI_LITE_ADDR_W(AXI_LITE_ADDR_W),
        .AXI_LITE_DATA_W(AXI_LITE_DATA_W),
        .AXIS_TDATA_W(AXIS_TDATA_W),
        .AXI_FULL_ADDR_W(AXI_FULL_ADDR_W),
        .AXI_FULL_DATA_W(AXI_FULL_DATA_W),
        .IMG_W(IMG_W),
        .IMG_H(IMG_H)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tuser(s_axis_tuser),
        .s_axis_tlast(s_axis_tlast),
        .m0_axi_awaddr(m0_axi_awaddr),
        .m0_axi_awlen(m0_axi_awlen),
        .m0_axi_awsize(m0_axi_awsize),
        .m0_axi_awburst(m0_axi_awburst),
        .m0_axi_awvalid(m0_axi_awvalid),
        .m0_axi_awready(m0_axi_awready),
        .m0_axi_wdata(m0_axi_wdata),
        .m0_axi_wstrb(m0_axi_wstrb),
        .m0_axi_wlast(m0_axi_wlast),
        .m0_axi_wvalid(m0_axi_wvalid),
        .m0_axi_wready(m0_axi_wready),
        .m0_axi_bresp(m0_axi_bresp),
        .m0_axi_bvalid(m0_axi_bvalid),
        .m0_axi_bready(m0_axi_bready),
        .m1_axi_awaddr(m1_axi_awaddr),
        .m1_axi_awlen(m1_axi_awlen),
        .m1_axi_awsize(m1_axi_awsize),
        .m1_axi_awburst(m1_axi_awburst),
        .m1_axi_awvalid(m1_axi_awvalid),
        .m1_axi_awready(m1_axi_awready),
        .m1_axi_wdata(m1_axi_wdata),
        .m1_axi_wstrb(m1_axi_wstrb),
        .m1_axi_wlast(m1_axi_wlast),
        .m1_axi_wvalid(m1_axi_wvalid),
        .m1_axi_wready(m1_axi_wready),
        .m1_axi_bresp(m1_axi_bresp),
        .m1_axi_bvalid(m1_axi_bvalid),
        .m1_axi_bready(m1_axi_bready),
        .m1_axi_araddr(m1_axi_araddr),
        .m1_axi_arlen(m1_axi_arlen),
        .m1_axi_arsize(m1_axi_arsize),
        .m1_axi_arburst(m1_axi_arburst),
        .m1_axi_arvalid(m1_axi_arvalid),
        .m1_axi_arready(m1_axi_arready),
        .m1_axi_rdata(m1_axi_rdata),
        .m1_axi_rlast(m1_axi_rlast),
        .m1_axi_rresp(m1_axi_rresp),
        .m1_axi_rvalid(m1_axi_rvalid),
        .m1_axi_rready(m1_axi_rready),
        .irq(irq)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    // ==========================================
    // Robust AXI Memory Model (Writes)
    // ==========================================
    assign m0_axi_awready = 1'b1;
    assign m0_axi_wready  = 1'b1;
    assign m0_axi_bvalid  = 1'b1;
    assign m0_axi_bresp   = 2'b00;

    assign m1_axi_awready = 1'b1;
    assign m1_axi_wready  = 1'b1;
    assign m1_axi_bvalid  = 1'b1;
    assign m1_axi_bresp   = 2'b00;

    logic m1_aw_pending;
    logic m1_w_pending;
    logic [AXI_FULL_ADDR_W-1:0] m1_awaddr_reg;
    logic [AXI_FULL_DATA_W-1:0] m1_wdata_reg;
    logic [7:0] m1_wstrb_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1_aw_pending <= 1'b0;
            m1_w_pending  <= 1'b0;
        end else begin
            if (m0_axi_wvalid && m0_axi_wready && m0_axi_awvalid) begin
                for (int i = 0; i < AXI_FULL_DATA_W/8; i++) begin
                    if (m0_axi_wstrb[i]) ram[m0_axi_awaddr + i] <= m0_axi_wdata[i*8 +: 8];
                end
            end
            
            if (m1_axi_awvalid && m1_axi_awready) begin
                m1_aw_pending <= 1'b1;
                m1_awaddr_reg <= m1_axi_awaddr;
            end
            if (m1_axi_wvalid && m1_axi_wready) begin
                m1_w_pending <= 1'b1;
                m1_wdata_reg <= m1_axi_wdata;
                m1_wstrb_reg <= m1_axi_wstrb;
            end
            
            if ((m1_aw_pending || (m1_axi_awvalid && m1_axi_awready)) &&
                (m1_w_pending || (m1_axi_wvalid && m1_axi_wready))) begin
                for (int i = 0; i < AXI_FULL_DATA_W/8; i++) begin
                    if (m1_w_pending ? m1_wstrb_reg[i] : m1_axi_wstrb[i]) begin
                        ram[(m1_aw_pending ? m1_awaddr_reg : m1_axi_awaddr) + i] <= m1_w_pending ? m1_wdata_reg[i*8 +: 8] : m1_axi_wdata[i*8 +: 8];
                    end
                end
                m1_aw_pending <= 1'b0;
                m1_w_pending  <= 1'b0;
            end
        end
    end

    // ==========================================
    // AXI Memory Model (Reads)
    // ==========================================
    logic [7:0] rlen_cnt;
    logic       reading;
    logic [AXI_FULL_ADDR_W-1:0] raddr;

    assign m1_axi_arready = !reading;
    assign m1_axi_rresp   = 2'b00;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reading <= 0;
            m1_axi_rvalid <= 0;
            m1_axi_rlast <= 0;
        end else begin
            if (m1_axi_arvalid && m1_axi_arready) begin
                reading <= 1;
                raddr   <= m1_axi_araddr;
                rlen_cnt <= m1_axi_arlen;
                m1_axi_rvalid <= 1;
                m1_axi_rlast  <= (m1_axi_arlen == 0);
            end else if (reading && m1_axi_rvalid && m1_axi_rready) begin
                if (rlen_cnt == 0) begin
                    reading <= 0;
                    m1_axi_rvalid <= 0;
                    m1_axi_rlast <= 0;
                end else begin
                    rlen_cnt <= rlen_cnt - 1;
                    raddr    <= raddr + (AXI_FULL_DATA_W/8);
                    m1_axi_rlast <= (rlen_cnt == 1);
                end
            end
        end
    end

    always_comb begin
        m1_axi_rdata = '0;
        if (reading) begin
            for (int i = 0; i < AXI_FULL_DATA_W/8; i++) begin
                m1_axi_rdata[i*8 +: 8] = ram[raddr + i];
            end
        end
    end

    // ==========================================
    // BULLETPROOF FSM & PROGRESS TRACKER
    // ==========================================
    logic [2:0] prev_state;
    logic [1:0] prev_rstate;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            prev_state <= 3'b000;
            prev_rstate <= 2'b00;
        end else begin
            // 1. Trace FSM Transitions reliably
            if (dut.u_backend_processing.state != prev_state) begin
                $display("[%0t] [FSM-TRACE] Main State changed: %0d -> %0d", $time, prev_state, dut.u_backend_processing.state);
                prev_state <= dut.u_backend_processing.state;
            end
            
            // 2. Trace DMA Inner State
            if (dut.u_backend_processing.state == 3'b011) begin
                if (dut.u_backend_processing.p2_rstate != prev_rstate) begin
                    prev_rstate <= dut.u_backend_processing.p2_rstate;
                end
                
                // 3. Trace DMA Pixel Progress (Every 250,000 pixels)
                if (dut.u_backend_processing.p2_stream_tvalid && dut.u_backend_processing.p2_stream_tlast) begin
                    if ((dut.u_backend_processing.p2_pixel_cnt / IMG_W) % 100 == 0) begin
                        $display("[%0t] [DMA-PROGRESS] Fetched %0d / %0d total lines from DDR", 
                                 $time, dut.u_backend_processing.p2_pixel_cnt / IMG_W, IMG_H);
                    end
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            if ($rose(dut.u_backend_processing.pass1_done))    
                $display("[%0t] STAGE: CCL Pass 1 Completed", $time);
            if ($rose(dut.u_backend_processing.resolver_done)) 
                $display("[%0t] STAGE: CCL Resolver Completed", $time);
            if ($rose(dut.u_backend_processing.stats_done))    
                $display("[%0t] STAGE: CCL Stats & Pass 2 Completed", $time);
            if ($rose(dut.u_backend_processing.geo_done))      
                $display("[%0t] STAGE: Geometry Filter Completed | Best Label: %0d", $time, dut.u_backend_processing.best_label);
            if ($rose(dut.u_backend_processing.fetch_done))    
                $display("[%0t] STAGE: ROI Fetch Completed", $time);
            if ($rose(dut.u_backend_processing.match_done))    
                $display("[%0t] STAGE: Template Matching Completed", $time);
        end
    end

    task axi_write(input [11:0] addr, input [31:0] data);
        int timeout_cnt = 0;
        $display("[%0t]   -> AXI Write Task: Initiating Addr=%03x, Data=%08x", $time, addr, data);
        @(posedge clk);
        #1; 
        s_axi_awaddr  = addr;
        s_axi_awvalid = 1'b1;
        s_axi_wdata   = data;
        s_axi_wvalid  = 1'b1;
        s_axi_wstrb   = 4'hF;

        while (s_axi_awvalid || s_axi_wvalid) begin
            @(posedge clk);
            #1;
            if (s_axi_awvalid && s_axi_awready) s_axi_awvalid = 1'b0;
            if (s_axi_wvalid && s_axi_wready)   s_axi_wvalid  = 1'b0;
        end
        
        $display("[%0t]   -> AXI Write Task: AW and W channels completed. Waiting for BVALID...", $time);
        
        timeout_cnt = 0;
        while (!s_axi_bvalid && timeout_cnt < 20) begin
            @(posedge clk);
            #1;
            timeout_cnt++;
        end
        
        @(posedge clk);
        #1;
    endtask

    initial begin
        #4000000000;
        $display("[%0t] ERROR: Watchdog Timeout! Simulation is stuck.", $time);
        $finish;
    end

    initial begin
        clk = 0;
        rst_n = 0;
        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        s_axis_tuser = 0;
        s_axis_tlast = 0;
        s_axi_awaddr = 0;
        s_axi_awvalid = 0;
        s_axi_wdata = 0;
        s_axi_wstrb = 4'hF;
        s_axi_wvalid = 0;
        s_axi_bready = 1; 
        s_axi_araddr = 0;
        s_axi_arvalid = 0;
        s_axi_rready = 1;

        fd_in         = $fopen("design/work/ProjectA/data/image_in.hex", "r");
        fd_mask_out   = $fopen("design/work/ProjectA/data/mask_output.hex", "w");
        fd_morph_out  = $fopen("design/work/ProjectA/data/morph_output.hex", "w");
        fd_labels_out = $fopen("design/work/ProjectA/data/labels_output.hex", "w");
        fd_geom_out   = $fopen("design/work/ProjectA/data/geom_bboxes.txt", "w");
        fd_match_out  = $fopen("design/work/ProjectA/data/template_matches.txt", "w");

        if (fd_in == 0) begin
            $display("[%0t] Error: Could not open input_image.hex", $time);
            $finish;
        end

        $display("[%0t] Applying Reset...", $time);
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 10);

        $display("[%0t] Configuring Base Address (Reg 0x14)...", $time);
        axi_write(12'h014, 32'h00000000);
        
        $display("[%0t] Enabling System (Reg 0x00)...", $time);
        axi_write(12'h000, 32'h00000001);

        #(CLK_PERIOD * 10);

        $display("[%0t] Starting Image Stream Injection (Resolution: %0dx%0d)...", $time, IMG_W, IMG_H);
        for (int y = 0; y < IMG_H; y++) begin
            if (y % 100 == 0) $display("[%0t] Injecting line %0d / %0d", $time, y, IMG_H);
            
            for (int x = 0; x < IMG_W; x++) begin
                if (!$feof(fd_in)) begin
                    $fscanf(fd_in, "%h\n", s_axis_tdata);
                    s_axis_tvalid = 1'b1;
                    s_axis_tuser  = (x == 0 && y == 0) ? 1'b1 : 1'b0;
                    s_axis_tlast  = (x == IMG_W - 1) ? 1'b1 : 1'b0;
                    
                    @(posedge clk);
                    while (!s_axis_tready) begin
                        @(posedge clk);
                    end
                end else begin
                    break;
                end
            end
        end

        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
        s_axis_tuser  <= 1'b0;
        s_axis_tdata  <= '0;

        $display("[%0t] Image Injection Complete. Waiting for IRQ...", $time);
        
        wait(irq == 1'b1);
        $display("[%0t] IRQ Received! Processing Complete.", $time);
        
        #(CLK_PERIOD * 50);

        $fclose(fd_in);
        $fclose(fd_mask_out);
        $fclose(fd_morph_out);
        $fclose(fd_labels_out);
        $fclose(fd_geom_out);
        $fclose(fd_match_out);
        
        $display("[%0t] Simulation Complete.", $time);
        $finish;
    end

    // Dump signals locally
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (dut.stream_mask_tvalid) $fdisplay(fd_mask_out, "%h", dut.stream_mask_tdata);
            if (dut.stream_morph_tvalid) $fdisplay(fd_morph_out, "%h", dut.stream_morph_tdata);
            if (dut.u_backend_processing.u_ccl_stats.s_axis_tvalid) $fdisplay(fd_labels_out, "%04x", dut.u_backend_processing.u_ccl_stats.s_axis_label);
            if (dut.u_backend_processing.u_geometry_filter.obj_valid) begin
                $fdisplay(fd_geom_out, "LBL: %d | XMIN: %d, XMAX: %d, YMIN: %d, YMAX: %d", 
                          dut.u_backend_processing.u_geometry_filter.obj_label,
                          dut.u_backend_processing.u_geometry_filter.obj_xmin,
                          dut.u_backend_processing.u_geometry_filter.obj_xmax,
                          dut.u_backend_processing.u_geometry_filter.obj_ymin,
                          dut.u_backend_processing.u_geometry_filter.obj_ymax);
            end
            if (dut.u_backend_processing.u_matcher.match_done) begin
                $fdisplay(fd_match_out, "Match Complete | Class ID: %d | Best Score: %d",
                          dut.u_backend_processing.u_matcher.best_class_id,
                          dut.u_backend_processing.u_matcher.best_score);
            end
        end
    end
endmodule