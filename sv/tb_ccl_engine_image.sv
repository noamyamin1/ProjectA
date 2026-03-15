`timescale 1ns / 1ps

module tb_ccl_engine_image;

    localparam int IMG_WIDTH  = 2872;
    localparam int IMG_HEIGHT = 1617;
    localparam int TOTAL_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam int LABEL_W    = 11; 
    
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

    int fd_in;
    int fd_out;
    logic [7:0] char_val;
    
    // Internal buffer for Pass 1 temporary labels
    logic [LABEL_W-1:0] pass1_frame_buffer [0:TOTAL_PIXELS-1];
    int pixel_capture_cnt;

    // Snooping Pass 1 outputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_capture_cnt <= 0;
        end else if (rst_n && dut.pass1_tvalid) begin
            pass1_frame_buffer[pixel_capture_cnt] <= dut.pass1_tdata;
            pixel_capture_cnt <= pixel_capture_cnt + 1;
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

        fd_in = $fopen("/users/epnyrk/Project/design/work/ProjectA/sv/morph_out.txt", "r");
        if (!fd_in) begin
            $display("Error opening morph_out.txt");
            $finish;
        end

        #20 rst_n = 1;
        #10;

        // Stream pixels to CCL
        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            
            void'($fscanf(fd_in, "%c\n", char_val));
            
            if (char_val == "1") s_axis_tdata <= 1'b1;
            else                 s_axis_tdata <= 1'b0;
            
            s_axis_tuser <= (i == 0);
            s_axis_tlast <= ((i + 1) % IMG_WIDTH == 0);
        end
        
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        s_axis_tuser  <= 1'b0;
        s_axis_tlast  <= 1'b0;
        $fclose(fd_in);

        // Wait for Union-Find Resolver
        wait(resolver_done);
        $display("Resolver finished! Starting Pass 2 (TB emulation)...");
        
        // Pass 2 Simulation 
        fd_out = $fopen("/users/epnyrk/Project/design/work/ProjectA/sv/ccl_labels_out.txt", "w");
        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            p2_parent_addr = pass1_frame_buffer[i]; 
            
            @(posedge clk); // Setup
            @(posedge clk); // Read
            
            $fdisplay(fd_out, "%0d", p2_parent_rdata);
        end
        
        $fclose(fd_out);
        $display("Simulation complete. Wrote final labels to ccl_labels_out.txt");
        $finish;
    end
endmodule