`timescale 1ns / 1ps

module tb_geo_subsystem;

    localparam int IMG_WIDTH  = 2872;
    localparam int IMG_HEIGHT = 1617;
    localparam int TOTAL_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam int LABEL_W    = 16;
    
    logic clk;
    logic rst_n;
    
    // Stats Collector Inputs
    logic [LABEL_W-1:0] s_axis_label;
    logic               s_axis_tvalid;
    logic               s_axis_tuser;
    logic               s_axis_tlast;
    
    // Stats <-> Dummy Parent RAM 
    logic [LABEL_W-1:0] parent_addr;
    logic [LABEL_W-1:0] parent_rdata;
    
    // Stats <-> Filter Interface
    logic [LABEL_W-1:0] geo_ram_addr;
    logic [31:0]        ram_area;
    logic [31:0]        ram_perimeter;
    logic [15:0]        ram_xmin;
    logic [15:0]        ram_xmax;
    logic [15:0]        ram_ymin;
    logic [15:0]        ram_ymax;
    logic               stats_done;
    
    // Geometry Filter Interface
    logic               filter_start;
    logic [LABEL_W-1:0] max_label_sim;
    logic               filter_done;
    logic               obj_valid;
    logic [LABEL_W-1:0] obj_label;
    logic [15:0]        obj_xmin;
    logic [15:0]        obj_xmax;
    logic [15:0]        obj_ymin;
    logic [15:0]        obj_ymax;

    // Instantiations
    ccl_stats_collector #(
        .LABEL_W(LABEL_W),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) dut_stats (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_label(s_axis_label),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tuser(s_axis_tuser),
        .s_axis_tlast(s_axis_tlast),
        .parent_addr(parent_addr),
        .parent_rdata(parent_rdata),
        .geo_ram_addr(geo_ram_addr),
        .out_area(ram_area),
        .out_perimeter(ram_perimeter),
        .out_xmin(ram_xmin),
        .out_xmax(ram_xmax),
        .out_ymin(ram_ymin),
        .out_ymax(ram_ymax),
        .stats_done(stats_done)
    );

    geometry_filter #(
        .LABEL_W(LABEL_W),
        .MIN_AREA_TH(1000),
        .MAX_AREA_TH(100000)
    ) dut_filter (
        .clk(clk),
        .rst_n(rst_n),
        .start(filter_start),
        .max_label(max_label_sim),
        .ram_addr(geo_ram_addr),
        .ram_area(ram_area),
        .ram_perimeter(ram_perimeter),
        .ram_xmin(ram_xmin),
        .ram_xmax(ram_xmax),
        .ram_ymin(ram_ymin),
        .ram_ymax(ram_ymax),
        .filter_done(filter_done),
        .obj_valid(obj_valid),
        .obj_label(obj_label),
        .obj_xmin(obj_xmin),
        .obj_xmax(obj_xmax),
        .obj_ymin(obj_ymin),
        .obj_ymax(obj_ymax)
    );

    assign parent_rdata = parent_addr;

    always #5 clk = ~clk;

    int fd_in;
    int fd_out;
    logic [7:0] char_val [0:10]; 
    int read_code;

    int bbox_width;
    int bbox_height;
    int spatial_area;
    int pixel_mass;
    real fill_ratio;

    always_ff @(posedge clk) begin
        if (obj_valid) begin
            // Matching SW model dimensions calculation
            bbox_width   = obj_xmax - obj_xmin;
            bbox_height  = obj_ymax - obj_ymin;
            spatial_area = bbox_width * bbox_height;
            pixel_mass   = ram_area; // Assuming ram_area holds the pixel count from CCL stats
            
            if (spatial_area > 0) begin
                fill_ratio = real'(pixel_mass) / real'(spatial_area);
            end else begin
                fill_ratio = 0.0;
            end

            // Write to CSV: label, xmin, ymin, xmax, ymax, w, h, spatial_area, pixel_mass, fill_ratio
            $fdisplay(fd_out, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%f", 
                      obj_label, obj_xmin, obj_ymin, obj_xmax, obj_ymax, 
                      bbox_width, bbox_height, spatial_area, pixel_mass, fill_ratio);
        end
    end

    // Watchdog Timer to prevent infinite hangs
    initial begin
        #50ms; 
        $display("[%0t] FATAL ERROR: Watchdog timeout reached. Simulation is stuck.", $time);
        $finish;
    end

    initial begin
        int tmp_label;
        
        clk = 0;
        rst_n = 0;
        s_axis_tvalid = 0;
        filter_start = 0;
        max_label_sim = 0;

        fd_out = $fopen("design/work/ProjectA/sv/detected_boxes.txt", "w");
        if (fd_out == 0) begin
            $display("FATAL ERROR: Could not open detected_boxes.txt for writing.");
            $finish;
        end

        #20 rst_n = 1;
        #10;
        
        $display("[%0t] Waiting for Stats RAM Initialization...", $time);
        wait(dut_stats.init_en == 1'b0);
        $display("[%0t] Stats RAM Initialized.", $time);

        fd_in = $fopen("design/work/ProjectA/sv/ccl_labels_out.txt", "r");
        if (fd_in == 0) begin
            $display("FATAL ERROR: Could not open ccl_labels_out.txt for reading.");
            $finish;
        end
        
        $display("[%0t] Starting Pixel Stream (%0d pixels)...", $time, TOTAL_PIXELS);
        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            @(posedge clk);
            s_axis_tvalid <= 1'b1;
            
            read_code = $fscanf(fd_in, "%d", tmp_label);
            if (read_code == 1) begin
                s_axis_label <= tmp_label;
            end else begin
                s_axis_label <= '0;
            end
            
            if (s_axis_label > max_label_sim) max_label_sim <= s_axis_label;
            
            s_axis_tuser <= (i == 0);
            s_axis_tlast <= ((i + 1) % IMG_WIDTH == 0);
        end
        
        @(posedge clk);
        s_axis_tvalid <= 1'b0;
        $fclose(fd_in);
        $display("[%0t] Pixel Stream Finished. Waiting for stats_done...", $time);

        wait(stats_done);
        $display("[%0t] stats_done asserted. Triggering Geometry Filter...", $time);
        
        @(posedge clk);
        filter_start <= 1'b1;
        
        @(posedge clk);
        filter_start <= 1'b0;
        
        $display("[%0t] Waiting for filter_done...", $time);
        wait(filter_done);
        
        $display("[%0t] filter_done asserted. Simulation Finished Successfully.", $time);
        $fclose(fd_out);
        $finish;
    end

endmodule