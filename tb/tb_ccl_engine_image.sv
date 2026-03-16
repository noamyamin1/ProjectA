`timescale 1ns / 1ps

module tb_ccl_engine_debug;

    localparam int IMG_WIDTH  = 2872;
    localparam int IMG_HEIGHT = 1617;
    localparam int TOTAL_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam int LABEL_W = 16; 
    
    logic clk;
    logic rst_n;
    
    logic s_axis_tdata;
    logic s_axis_tvalid;
    logic s_axis_tuser;
    logic s_axis_tlast;
    
    logic pass1_done;
    logic resolver_done;
    
    logic [LABEL_W-1:0] p2_parent_addr;
    logic [LABEL_W-1:0] p2_parent_rdata;

    ccl_engine #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .LABEL_W(LABEL_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tuser(s_axis_tuser),
        .s_axis_tlast(s_axis_tlast),
        .pass1_done(pass1_done),
        .resolver_done(resolver_done),
        .p2_parent_addr(p2_parent_addr),
        .p2_parent_rdata(p2_parent_rdata)
    );

    always #5 clk = ~clk;

    // ==========================================
    // DIAGNOSTIC SNOOPER: Saturation Monitor
    // ==========================================
    always_ff @(posedge clk) begin
        if (rst_n && dut.u_pass1.next_label_cnt == {LABEL_W{1'b1}}) begin
            $display("CRITICAL WARNING: Label Saturation Reached at time %0t! Objects will be dropped to 0.", $time);
        end
    end

    int fd_in, fd_out, fd_raw, fd_log;
    logic [7:0] char_val;
    
    logic [LABEL_W-1:0] pass1_frame_buffer [0:TOTAL_PIXELS-1];
    int pixel_capture_cnt = 0;

    always_ff @(posedge clk) begin
        if (rst_n && dut.pass1_tvalid) begin
            pass1_frame_buffer[pixel_capture_cnt] <= dut.pass1_tdata;
            pixel_capture_cnt <= pixel_capture_cnt + 1;
        end
    end

    // ==========================================
    // DIAGNOSTIC SNOOPER: Log all RAM updates
    // ==========================================
    initial begin
        fd_log = $fopen("ccl_ram_events.log", "w");
    end
    always_ff @(posedge clk) begin
        if (rst_n && dut.ram_we) begin
            $fdisplay(fd_log, "TIME: %0t | RAM WRITE | Addr: %0d <= Data: %0d", $time, dut.ram_addr, dut.ram_wdata);
        end
    end

    // ==========================================
    // MAIN STIMULUS
    // ==========================================
    initial begin
        clk = 0;
        rst_n = 0;
        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        s_axis_tuser = 0;
        s_axis_tlast = 0;
        p2_parent_addr = 0;

        fd_in = $fopen("/users/epnyrk/Project/design/work/ProjectA/sv/morph_out.txt", "r");
        #20 rst_n = 1;
        #10;

        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            void'($fscanf(fd_in, "%s", char_val));
            s_axis_tdata <= (char_val == "1");
            s_axis_tuser <= (i == 0);
            s_axis_tlast <= ((i + 1) % IMG_WIDTH == 0);
        end
        
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tuser  <= 1'b0;
        s_axis_tlast  <= 1'b0;
        $fclose(fd_in);

        // DIAGNOSTIC DUMP: Pass 1 Raw Labels (Before Resolver modifies RAM)
        fd_raw = $fopen("ccl_pass1_raw.txt", "w");
        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            $fdisplay(fd_raw, "%0d", pass1_frame_buffer[i]);
        end
        $fclose(fd_raw);
        $display("Dumped Raw Pass 1 Labels.");

        wait(resolver_done);
        $display("Resolver finished! Starting Pass 2 (TB emulation)...");
        
        fd_out = $fopen("ccl_labels_out.txt", "w");
        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            p2_parent_addr = pass1_frame_buffer[i]; 
            @(posedge clk); 
            @(posedge clk); 
            $fdisplay(fd_out, "%0d", p2_parent_rdata);
        end
        
        $fclose(fd_out);
        $fclose(fd_log);
        $display("Simulation complete.");
        $finish;
    end
endmodule