`timescale 1ns / 1ps

module tb_red_mask_datapath_image;

    localparam int TDATA_W = 24;
    localparam int IMG_WIDTH = 1920;
    localparam int IMG_HEIGHT = 1080;
    localparam int TOTAL_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    
    logic clk;
    logic rst_n;
    
    logic [7:0] min_red_val;
    logic [2:0] margin_shift;
    
    logic [TDATA_W-1:0] s_axis_tdata;
    logic s_axis_tvalid;
    logic s_axis_tready;
    logic s_axis_tuser;
    logic s_axis_tlast;
    
    logic m_axis_tdata;
    logic m_axis_tvalid;
    logic m_axis_tready;
    logic m_axis_tuser;
    logic m_axis_tlast;

    red_mask_datapath #(
        .TDATA_W(TDATA_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .min_red_val(min_red_val),
        .margin_shift(margin_shift),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tuser(s_axis_tuser),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tuser(m_axis_tuser),
        .m_axis_tlast(m_axis_tlast)
    );

    always #5 clk = ~clk;

    int fd_in;
    int fd_out;
    
    initial begin
        clk = 0;
        rst_n = 0;
        min_red_val = 8'd15;
        margin_shift = 3'd3;
        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        s_axis_tuser = 0;
        s_axis_tlast = 0;
        m_axis_tready = 1'b1;

        fd_in = $fopen("design/work/ProjectA/data/image_in.hex", "r");
        if (!fd_in) begin
            $display("ERROR: Could not open image_in.hex in data directory");
            $finish;
        end

        fd_out = $fopen("design/work/ProjectA/data/actual_mask_out.txt", "w");
        if (!fd_out) begin
            $display("ERROR: Could not open actual_mask_out.txt in data directory");
            $finish;
        end

        #20 rst_n = 1;
        #10;

        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            
            void'($fscanf(fd_in, "%h\n", s_axis_tdata));
            
            s_axis_tuser <= (i == 0);
            s_axis_tlast <= ((i + 1) % IMG_WIDTH == 0);
            
            do begin
                @(posedge clk);
            end while (!s_axis_tready);
            
            s_axis_tvalid <= 1'b0;
            s_axis_tuser  <= 1'b0;
            s_axis_tlast  <= 1'b0;
        end

        $fclose(fd_in);
    end

    initial begin
        int pixels_received = 0;
        
        forever begin
            @(posedge clk);
            if (m_axis_tvalid && m_axis_tready) begin
                $fdisplay(fd_out, "%b", m_axis_tdata);
                pixels_received++;
                
                if (pixels_received == TOTAL_PIXELS) begin
                    $display("Simulation complete. Processed %0d pixels.", pixels_received);
                    $fclose(fd_out);
                    $finish;
                end
            end
        end
    end

endmodule