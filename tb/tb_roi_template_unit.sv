`timescale 1ns / 1ps

module tb_roi_template_unit;

    localparam int IMG_W = 1920;
    localparam int IMG_H = 1080;
    localparam int TOTAL_PIXELS = IMG_W * IMG_H;
    localparam int MAX_ROIS = 256;
    localparam int TEMPLATE_COUNT = 18;

    logic clk;
    logic rst_n;

    logic start;
    logic [31:0] base_addr;
    logic [15:0] roi_xmin;
    logic [15:0] roi_xmax;
    logic [15:0] roi_ymin;
    logic [15:0] roi_ymax;

    logic [31:0] m_axi_araddr;
    logic [7:0]  m_axi_arlen;
    logic [2:0]  m_axi_arsize;
    logic [1:0]  m_axi_arburst;
    logic        m_axi_arvalid;
    logic        m_axi_arready;

    logic [63:0] m_axi_rdata;
    logic [1:0]  m_axi_rresp;
    logic        m_axi_rlast;
    logic        m_axi_rvalid;
    logic        m_axi_rready;

    logic [7:0]  m_axis_gray_tdata;
    logic        m_axis_gray_tvalid;
    logic        m_axis_gray_tlast;
    logic        fetch_done;

    logic [10:0] template_ram_addr;
    logic [31:0] template_ram_rdata;
    logic        match_done;
    logic [7:0]  best_class_id;
    logic [31:0] best_score;

    roi_fetcher_axi_master #(
        .AXI_ADDR_W(32),
        .AXI_DATA_W(64),
        .FRAME_W(IMG_W)
    ) u_roi_fetcher (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .base_addr(base_addr),
        .roi_xmin(roi_xmin),
        .roi_xmax(roi_xmax),
        .roi_ymin(roi_ymin),
        .roi_ymax(roi_ymax),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axis_gray_tdata(m_axis_gray_tdata),
        .m_axis_gray_tvalid(m_axis_gray_tvalid),
        .m_axis_gray_tlast(m_axis_gray_tlast),
        .fetch_done(fetch_done)
    );

    template_matching_engine #(
        .TEMPLATE_COUNT(TEMPLATE_COUNT)
    ) u_matcher (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_gray_tdata(m_axis_gray_tdata),
        .s_axis_gray_tvalid(m_axis_gray_tvalid),
        .s_axis_gray_tlast(m_axis_gray_tlast),
        .template_ram_addr(template_ram_addr),
        .template_ram_rdata(template_ram_rdata),
        .match_done(match_done),
        .best_class_id(best_class_id),
        .best_score(best_score)
    );

    always #5 clk = ~clk;

    logic [31:0] image_mem [0:TOTAL_PIXELS-1];
    logic [31:0] template_rom [0:1023];
    int roi_count;
    int roi_ids [0:MAX_ROIS-1];
    int roi_xmin_arr [0:MAX_ROIS-1];
    int roi_xmax_arr [0:MAX_ROIS-1];
    int roi_ymin_arr [0:MAX_ROIS-1];
    int roi_ymax_arr [0:MAX_ROIS-1];

    int fd_img;
    int fd_roi;
    int fd_out;
        int fd_bin;
    int cycle_cnt;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        base_addr = 32'h0000_0000;
        roi_xmin = 0;
        roi_xmax = 0;
        roi_ymin = 0;
        roi_ymax = 0;
        roi_count = 0;

        fd_img = $fopen("design/work/ProjectA/data/image_in.hex", "r");
        if (!fd_img) begin
            $display("ERROR: Could not open image_in.hex");
            $finish;
        end

        for (int i = 0; i < TOTAL_PIXELS; i++) begin
            logic [23:0] pixel24;
            void'($fscanf(fd_img, "%h\n", pixel24));
            image_mem[i] = {8'h00, pixel24};
        end
        $fclose(fd_img);

        $readmemh("design/work/ProjectA/data/templates.mem", template_rom);

        fd_roi = $fopen("design/work/ProjectA/data/roi_list.txt", "r");
        if (!fd_roi) begin
            $display("ERROR: Could not open roi_list.txt");
            $finish;
        end

        while (!$feof(fd_roi)) begin
            string line;
            int id, xmin, xmax, ymin, ymax, label;
            line = "";
            if ($fgets(line, fd_roi) == 0) begin
                break;
            end
            if (line.len() == 0 || line.substr(0, 0) == "#") begin
                continue;
            end
            if ($sscanf(line, "%d %d %d %d %d %d", id, xmin, xmax, ymin, ymax, label) == 6) begin
                if (roi_count < MAX_ROIS) begin
                    roi_ids[roi_count] = id;
                    roi_xmin_arr[roi_count] = xmin;
                    roi_xmax_arr[roi_count] = xmax;
                    roi_ymin_arr[roi_count] = ymin;
                    roi_ymax_arr[roi_count] = ymax;
                    roi_count++;
                end
            end
        end
        $fclose(fd_roi);

        #50;
        rst_n = 1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt <= 0;
        end else begin
            cycle_cnt <= cycle_cnt + 1;
            if ((cycle_cnt % 200000) == 0) begin
                $display("[%0t] TB progress: cycle=%0d", $time, cycle_cnt);
            end
        end
    end

    always_comb begin
        template_ram_rdata = template_rom[template_ram_addr];
    end

    assign m_axi_arready = 1'b1;

    logic pending_read;
    logic [31:0] araddr_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_read <= 1'b0;
            araddr_q <= '0;
            m_axi_rvalid <= 1'b0;
            m_axi_rdata <= '0;
            m_axi_rlast <= 1'b0;
            m_axi_rresp <= 2'b00;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                araddr_q <= m_axi_araddr;
                pending_read <= 1'b1;
            end

            if (pending_read && !m_axi_rvalid) begin
                int word_idx;
                int even_idx;
                word_idx = (araddr_q >> 2);
                even_idx = word_idx & ~1;
                if (even_idx < 0) even_idx = 0;
                if (even_idx + 1 >= TOTAL_PIXELS) even_idx = TOTAL_PIXELS - 2;

                m_axi_rdata <= {image_mem[even_idx + 1], image_mem[even_idx]};
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
            end

            if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                pending_read <= 1'b0;
            end

        end
    end

    initial begin
        fd_out = $fopen("design/work/ProjectA/data/actual_template_matching.txt", "w");
        if (!fd_out) begin
            $display("ERROR: Could not open actual_template_matching.txt");
            $finish;
        end

        wait (rst_n == 1'b1);

        for (int i = 0; i < roi_count; i++) begin
            roi_xmin = roi_xmin_arr[i];
            roi_xmax = roi_xmax_arr[i];
            roi_ymin = roi_ymin_arr[i];
            roi_ymax = roi_ymax_arr[i];

            $display("[%0t] ROI %0d start: xmin=%0d xmax=%0d ymin=%0d ymax=%0d", $time, i, roi_xmin, roi_xmax, roi_ymin, roi_ymax);
            start = 1'b1;
            begin : wait_fetch
                int c;
                for (c = 0; c < 2000000; c++) begin
                    @(posedge clk);
                    if (fetch_done) disable wait_fetch;
                end
                $display("ERROR: fetch_done timeout at ROI %0d", i);
                $finish;
            end
            start = 1'b0;

            $display("[%0t] ROI %0d fetch_done", $time, i);

            begin : wait_match
                int c;
                for (c = 0; c < 4000000; c++) begin
                    @(posedge clk);
                    if (match_done) disable wait_match;
                end
                $display("ERROR: match_done timeout at ROI %0d", i);
                $finish;
            end
            $fdisplay(fd_out, "%0d %0d %0d", roi_ids[i], best_class_id, best_score);

            $display("[%0t] ROI %0d match_done: class=%0d score=%0d", $time, i, best_class_id, best_score);
            // Debug prints removed

                if (i == 0) begin
                    fd_bin = $fopen("design/work/ProjectA/data/actual_roi_bin_0.txt", "w");
                    if (fd_bin) begin
                        for (int r = 0; r < 32; r++) begin
                            $fdisplay(fd_bin, "%08x", u_matcher.bin_roi[r]);
                        end
                        $fclose(fd_bin);
                    end else begin
                        $display("WARNING: Could not open actual_roi_bin_0.txt");
                    end

                begin
                    int top_id[0:2];
                    int top_score[0:2];
                    int s;
                    int t;
                    top_id[0] = -1; top_id[1] = -1; top_id[2] = -1;
                    top_score[0] = 32'h7FFFFFFF; top_score[1] = 32'h7FFFFFFF; top_score[2] = 32'h7FFFFFFF;
                    for (t = 0; t < TEMPLATE_COUNT; t++) begin
                        s = u_matcher.debug_scores[t];
                        if (s < top_score[0]) begin
                            top_score[2] = top_score[1]; top_id[2] = top_id[1];
                            top_score[1] = top_score[0]; top_id[1] = top_id[0];
                            top_score[0] = s; top_id[0] = t;
                        end else if (s < top_score[1]) begin
                            top_score[2] = top_score[1]; top_id[2] = top_id[1];
                            top_score[1] = s; top_id[1] = t;
                        end else if (s < top_score[2]) begin
                            top_score[2] = s; top_id[2] = t;
                        end
                    end

                    fd_bin = $fopen("design/work/ProjectA/data/actual_top3.txt", "w");
                    if (fd_bin) begin
                        $fdisplay(fd_bin, "%0d %0d %0d %0d %0d %0d %0d", roi_ids[i],
                                  top_id[0], top_score[0], top_id[1], top_score[1], top_id[2], top_score[2]);
                        $fclose(fd_bin);
                    end else begin
                        $display("WARNING: Could not open actual_top3.txt");
                    end

                    fd_bin = $fopen("design/work/ProjectA/data/actual_scores_full.txt", "w");
                    if (fd_bin) begin
                        for (t = 0; t < TEMPLATE_COUNT; t++) begin
                            $fdisplay(fd_bin, "%0d %0d", t, u_matcher.debug_scores[t]);
                        end
                        $fclose(fd_bin);
                    end else begin
                        $display("WARNING: Could not open actual_scores_full.txt");
                    end
                end
                end

            @(posedge clk);
        end

        $fclose(fd_out);
        $display("Simulation complete. ROIs processed: %0d", roi_count);
        $finish;
    end

endmodule
