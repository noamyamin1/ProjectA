`timescale 1ns / 1ps

module tb_ccl_engine_debug;

    localparam int IMG_WIDTH  = 1920;
    localparam int IMG_HEIGHT = 1080;
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

    logic [LABEL_W-1:0] p1_axis_tdata;
    logic               p1_axis_tvalid;
    logic               p1_axis_tlast;

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
        .p2_parent_rdata(p2_parent_rdata),
        .p1_axis_tdata(p1_axis_tdata),
        .p1_axis_tvalid(p1_axis_tvalid),
        .p1_axis_tlast(p1_axis_tlast)
    );

    always #5 clk = ~clk;

    always_ff @(posedge clk) begin
        if (rst_n && dut.u_pass1.next_label_cnt == {LABEL_W{1'b1}}) begin
            $display("CRITICAL WARNING: Label Saturation Reached at time %0t!", $time);
        end
    end

    int fd_in, fd_out_p1, fd_out_p2, fd_log;
    logic [7:0] char_val;
    
    logic [LABEL_W-1:0] pass1_frame_buffer [0:TOTAL_PIXELS-1];
    int pixel_capture_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_capture_cnt <= 0;
            for (int i=0; i<TOTAL_PIXELS; i++) begin
                pass1_frame_buffer[i] <= '0;
            end
        end else if (dut.pass1_tvalid && pixel_capture_cnt < TOTAL_PIXELS) begin
            pass1_frame_buffer[pixel_capture_cnt] <= dut.pass1_tdata;
            pixel_capture_cnt <= pixel_capture_cnt + 1;
        end
    end

    initial begin
        fd_log = $fopen("design/work/ProjectA/results/ccl_ram_events.log", "w");
    end
    
    always_ff @(posedge clk) begin
        if (rst_n && dut.ram_we) begin
            $fdisplay(fd_log, "TIME: %0t | RAM WRITE | Addr: %0d <= Data: %0d", $time, dut.ram_addr, dut.ram_wdata);
        end
    end

    initial begin
        clk = 0;
        rst_n = 0;
        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        s_axis_tuser = 0;
        s_axis_tlast = 0;
        p2_parent_addr = 0;

        fd_in = $fopen("design/work/ProjectA/data/actual_morph_out.txt", "r");
        if (!fd_in) begin
            $display("ERROR: Could not open actual_morph_out.txt");
            $finish;
        end

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

        wait(pixel_capture_cnt == TOTAL_PIXELS);

        fd_out_p1 = $fopen("design/work/ProjectA/data/actual_ccl_pass1.txt", "w");
        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            $fdisplay(fd_out_p1, "%0d", int'(pass1_frame_buffer[i]));
        end
        $fclose(fd_out_p1);
        $display("Dumped Raw Pass 1 Labels.");

        wait(resolver_done);
        $display("Resolver finished! Starting Pass 2 (TB emulation)...");
        
        fd_out_p2 = $fopen("design/work/ProjectA/data/actual_ccl_pass2.txt", "w");
        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            p2_parent_addr = pass1_frame_buffer[i]; 
            @(posedge clk); 
            @(posedge clk); 
            $fdisplay(fd_out_p2, "%0d", int'(p2_parent_rdata));
        end
        
        $fclose(fd_out_p2);
        $fclose(fd_log);
        $display("Simulation complete.");
        $finish;
    end
endmodule