`timescale 1ns / 1ps

module tb_backend_stats_geom;

    localparam int IMG_WIDTH = 1920;
    localparam int IMG_HEIGHT = 1080;
    localparam int TOTAL_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam int LABEL_W = 16;

    logic clk;
    logic rst_n;

    logic [LABEL_W-1:0] s_axis_label;
    logic s_axis_tvalid;
    logic s_axis_tuser;
    logic s_axis_tlast;

    logic [LABEL_W-1:0] parent_addr;
    logic [LABEL_W-1:0] parent_rdata;
    logic stats_clear;

    logic [LABEL_W-1:0] geo_ram_addr;
    logic [31:0] out_area;
    logic [31:0] out_perimeter;
    logic [15:0] out_xmin;
    logic [15:0] out_xmax;
    logic [15:0] out_ymin;
    logic [15:0] out_ymax;
    logic stats_done;
    logic stats_init_done;

    logic gf_start;
    logic [LABEL_W-1:0] gf_max_label;
    logic gf_done;

    logic gf_obj_valid;
    logic [LABEL_W-1:0] gf_obj_label;
    logic [15:0] gf_obj_xmin;
    logic [15:0] gf_obj_xmax;
    logic [15:0] gf_obj_ymin;
    logic [15:0] gf_obj_ymax;

`ifdef GEO_SOFT
    localparam int GEO_MIN_W = 35;
    localparam int GEO_MIN_H = 33;
    localparam int GEO_MIN_PIX_AREA = 350;
    localparam int GEO_MIN_FILL_NUM = 250;
    localparam int GEO_MIN_FILL_DEN = 1000;
    localparam int GEO_RELAX_SOL_NUM = 450;
    localparam int GEO_RELAX_SOL_DEN = 1000;
    localparam int GEO_MIN_W_RELAX = 31;
    localparam int GEO_MIN_H_RELAX = 30;
    localparam int GEO_ASPECT_RELAX_NUM = 22;
    localparam int GEO_ASPECT_RELAX_DEN = 10;
    localparam int GEO_MAX_CANDIDATES = 5;
`else
    localparam int GEO_MIN_W = 34;
    localparam int GEO_MIN_H = 32;
    localparam int GEO_MIN_PIX_AREA = 313;
    localparam int GEO_MIN_FILL_NUM = 218;
    localparam int GEO_MIN_FILL_DEN = 1000;
    localparam int GEO_RELAX_SOL_NUM = 400;
    localparam int GEO_RELAX_SOL_DEN = 1000;
    localparam int GEO_MIN_W_RELAX = 31;
    localparam int GEO_MIN_H_RELAX = 30;
    localparam int GEO_ASPECT_RELAX_NUM = 22;
    localparam int GEO_ASPECT_RELAX_DEN = 10;
    localparam int GEO_MAX_CANDIDATES = 5;
`endif

    // Use identity parent map so labels are treated as already-resolved pass2 labels.
    logic [LABEL_W-1:0] parent_mem [0:(1<<LABEL_W)-1];

    int fd_in;
    int fd_stats;
    int fd_boxes;
    logic [LABEL_W-1:0] in_lbl;
    int timeout_cycles;
    int init_cycles;

    always #5 clk = ~clk;

    always_comb begin
        parent_rdata = parent_mem[parent_addr];
    end

    ccl_stats_collector #(
        .LABEL_W(LABEL_W),
        .COORD_W(12),
        .MAX_COORD(4095),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_stats (
        .clk(clk),
        .rst_n(rst_n),
        .clear(stats_clear),
        .s_axis_label(s_axis_label),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tuser(s_axis_tuser),
        .s_axis_tlast(s_axis_tlast),
        .parent_addr(parent_addr),
        .parent_rdata(parent_rdata),
        .geo_ram_addr(geo_ram_addr),
        .out_area(out_area),
        .out_perimeter(out_perimeter),
        .out_xmin(out_xmin),
        .out_xmax(out_xmax),
        .out_ymin(out_ymin),
        .out_ymax(out_ymax),
        .stats_done(stats_done),
        .init_done(stats_init_done)
    );

    geometry_filter #(
        .LABEL_W(LABEL_W),
        .MIN_AREA_TH(300),
        .MAX_AREA_TH(100000),
        .MIN_W_TH(GEO_MIN_W),
        .MIN_H_TH(GEO_MIN_H),
        .MIN_PIX_AREA_TH(GEO_MIN_PIX_AREA),
        .FILL_MIN_NUM(GEO_MIN_FILL_NUM),
        .FILL_MIN_DEN(GEO_MIN_FILL_DEN),
        .MIN_W_RELAX_TH(GEO_MIN_W_RELAX),
        .MIN_H_RELAX_TH(GEO_MIN_H_RELAX),
        .RELAX_SOL_NUM(GEO_RELAX_SOL_NUM),
        .RELAX_SOL_DEN(GEO_RELAX_SOL_DEN),
        .ASPECT_RELAX_NUM(GEO_ASPECT_RELAX_NUM),
        .ASPECT_RELAX_DEN(GEO_ASPECT_RELAX_DEN),
        .MAX_CANDIDATES(GEO_MAX_CANDIDATES)
    ) u_geom (
        .clk(clk),
        .rst_n(rst_n),
        .start(gf_start),
        .max_label(gf_max_label),
        .ram_addr(geo_ram_addr),
        .ram_area(out_area),
        .ram_perimeter(out_perimeter),
        .ram_xmin(out_xmin),
        .ram_xmax(out_xmax),
        .ram_ymin(out_ymin),
        .ram_ymax(out_ymax),
        .filter_done(gf_done),
        .obj_valid(gf_obj_valid),
        .obj_label(gf_obj_label),
        .obj_xmin(gf_obj_xmin),
        .obj_xmax(gf_obj_xmax),
        .obj_ymin(gf_obj_ymin),
        .obj_ymax(gf_obj_ymax)
    );

    // Capture valid geometry-filter outputs.
    always_ff @(posedge clk) begin
        if (rst_n && gf_obj_valid) begin
            $fdisplay(fd_boxes, "%0d %0d %0d %0d %0d", gf_obj_label, gf_obj_xmin, gf_obj_xmax, gf_obj_ymin, gf_obj_ymax);
        end
    end

    task dump_stats_ram;
        int lbl;
        begin
            for (lbl = 1; lbl < (1<<LABEL_W); lbl = lbl + 1) begin
                if (u_stats.area_ram[lbl] != 0) begin
                    $fdisplay(fd_stats, "%0d %0d %0d %0d %0d %0d %0d",
                        lbl,
                        u_stats.area_ram[lbl],
                        u_stats.perim_ram[lbl],
                        u_stats.xmin_ram[lbl],
                        u_stats.xmax_ram[lbl],
                        u_stats.ymin_ram[lbl],
                        u_stats.ymax_ram[lbl]);
                end
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        s_axis_label = '0;
        s_axis_tvalid = 1'b0;
        s_axis_tuser = 1'b0;
        s_axis_tlast = 1'b0;
        stats_clear = 1'b0;
        gf_start = 1'b0;
        gf_max_label = 16'hFFFE;

        for (int i = 0; i < (1<<LABEL_W); i++) begin
            parent_mem[i] = i[LABEL_W-1:0];
        end

        fd_in = $fopen("design/work/ProjectA/data/actual_ccl_pass2.txt", "r");
        if (!fd_in) begin
            $display("ERROR: could not open data/actual_ccl_pass2.txt");
            $finish;
        end

        fd_stats = $fopen("design/work/ProjectA/data/actual_ccl_stats.txt", "w");
        fd_boxes = $fopen("design/work/ProjectA/data/actual_geom_boxes.txt", "w");

        $fdisplay(fd_stats, "# label area perimeter xmin xmax ymin ymax");
        $fdisplay(fd_boxes, "# label xmin xmax ymin ymax");

        #40;
        rst_n = 1'b1;
        #20;

        // Allow stats collector to finish RAM initialization (2^LABEL_W cycles).
        init_cycles = (1 << LABEL_W);
        repeat (init_cycles) @(posedge clk);

        // Stream all labels to stats collector.
        // Keep tuser low for this single-frame run.
        // In this RTL, tuser asserted during ST_PROCESS causes a state reset to ST_INIT.
        for (int pix = 0; pix < TOTAL_PIXELS; pix++) begin
            @(posedge clk);
            if ($fscanf(fd_in, "%d", in_lbl) != 1) begin
                $display("ERROR: insufficient input labels at pix=%0d", pix);
                $finish;
            end
            s_axis_label = in_lbl;
            s_axis_tvalid = 1'b1;
            s_axis_tuser = 1'b0;
            s_axis_tlast = (((pix + 1) % IMG_WIDTH) == 0);
        end

        @(posedge clk);
        s_axis_tvalid = 1'b0;
        s_axis_tuser = 1'b0;
        s_axis_tlast = 1'b0;

        $fclose(fd_in);

        timeout_cycles = 0;
        while (!stats_done && timeout_cycles < 5000000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end
        if (!stats_done) begin
            $display("ERROR: Timeout waiting for stats_done");
            $finish;
        end
        repeat (8) @(posedge clk);

        dump_stats_ram();
        $fclose(fd_stats);

        // Run geometry filter pass over collected RAM stats.
        // Hold start high through filtering to avoid handshake race/hang.
        @(posedge clk);
        gf_start <= 1'b1;

        timeout_cycles = 0;
        while (!gf_done && timeout_cycles < 5000000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end
        if (!gf_done) begin
            $display("ERROR: Timeout waiting for gf_done");
            $finish;
        end
        @(posedge clk);
        gf_start <= 1'b0;
        repeat (8) @(posedge clk);

        $fclose(fd_boxes);

        $display("tb_backend_stats_geom: completed");
        $finish;
    end

endmodule
