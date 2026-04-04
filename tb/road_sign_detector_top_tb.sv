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

    localparam int MAX_FRAMES = 64;
    localparam int MAX_PASSES = 2;
    localparam int NO_SIGN_SCORE_THRESHOLD = 230;
    localparam int TEMPLATE_COUNT = 18;
    localparam string PROJECT_A_DIR = "/users/epnyrk/Project/design/work/ProjectA";

    // ==========================================
    // File Handles & Status
    // ==========================================
    int f_mask_ref, f_morph_ref, f_ccl_ref, f_geom_ref, f_template_ref;
    int f_mask_actual, f_morph_actual, f_ccl_actual, f_ccl_pass2, f_ccl_resolved;
    int f_ccl_stats_e2e;
    int f_geom_filtered, f_rtl_roi, f_rtl_results, f_template_actual, f_scores_full, f_roi_bin;
    int f_summary;

    bit mask_ref_valid;
    bit morph_ref_valid;
    bit ccl_ref_valid;
    bit geom_ref_valid;
    bit template_ref_valid;

    int mask_errors;
    int morph_errors;
    int ccl_errors;
    int geom_errors;
    int geom_detected_count;
    int roi_pixel_cnt;
    int axi_errors;
    int template_errors;

    int pass_mask_errors [0:MAX_PASSES-1][0:MAX_FRAMES-1];
    int pass_morph_errors[0:MAX_PASSES-1][0:MAX_FRAMES-1];
    int pass_ccl_errors  [0:MAX_PASSES-1][0:MAX_FRAMES-1];
    int pass_geom_errors [0:MAX_PASSES-1][0:MAX_FRAMES-1];
    int pass_axi_errors  [0:MAX_PASSES-1][0:MAX_FRAMES-1];
    int pass_tmpl_errors [0:MAX_PASSES-1][0:MAX_FRAMES-1];
    int pass_total_errors[0:MAX_PASSES-1][0:MAX_FRAMES-1];

    integer frame_idx;

    int expected_best_id;
    int expected_best_score;
    bit expected_no_sign;
    logic [31:0] frame_best_score;
    logic [7:0]  frame_best_id;
    bit frame_best_valid;
    bit scores_dumped;
    bit roi_bin_dumped;

    // Array for fast image streaming
    logic [31:0] image_array [0:IMG_W*IMG_H-1];

    // Array for Template Descriptions
    string template_desc [0:19];

    // Frame control
    string golden_root;
    string data_root;
    string out_root;
    string frame_list [0:MAX_FRAMES-1];
    int frame_count;
    string mode;
    string single_frame;
    bit check_clear;
    bit hard_reset_between_frames;
    bit force_clear_between_frames;
    int current_pass;
    int current_frame;
    bit frame_active;
    string current_frame_name;
    string current_stem;
    int last_valid_class_id;

    // ==========================================
    // Signals
    // ==========================================
    logic clk = 0;
    logic rst_n = 0;

    // AXI4-Lite Slave Interface
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

    // AXI4-Stream Slave Interface
    logic [AXIS_TDATA_W-1:0]    s_axis_tdata = 0;
    logic                       s_axis_tvalid = 0;
    logic                       s_axis_tready;
    logic                       s_axis_tuser = 0;
    logic                       s_axis_tlast = 0;

    // AXI4 Master 0
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

    // AXI4 Master 1
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
    logic [7:0] current_id;
    road_sign_detector #(
        .AXI_LITE_ADDR_W(AXI_LITE_ADDR_W),
        .AXI_LITE_DATA_W(AXI_LITE_DATA_W),
        .AXIS_TDATA_W(AXIS_TDATA_W),
        .AXI_FULL_ADDR_W(AXI_FULL_ADDR_W),
        .AXI_FULL_DATA_W(AXI_FULL_DATA_W),
        .IMG_W(IMG_W),
        .IMG_H(IMG_H)
    ) dut (
        .current_id(current_id),
        .*
    );

    logic [31:0] final_score_debug;
    logic [7:0]  final_index_debug;

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
    logic [AXI_FULL_ADDR_W-1:0] m1_awaddr_latch;
    logic [AXI_FULL_DATA_W-1:0] m1_wdata_latch;
    logic [7:0]                 m1_wstrb_latch;
    logic                       m1_aw_pending;
    logic                       m1_w_pending;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1_axi_awready <= 1'b0;
            m1_axi_wready  <= 1'b0;
            m1_axi_bvalid  <= 1'b0;
            m1_axi_bresp   <= 2'b00;
            m1_awaddr_latch <= '0;
            m1_wdata_latch  <= '0;
            m1_wstrb_latch  <= '0;
            m1_aw_pending   <= 1'b0;
            m1_w_pending    <= 1'b0;
        end else begin
            m1_axi_awready <= 1'b1;
            m1_axi_wready  <= 1'b1;

            if (m1_axi_awvalid && m1_axi_awready) begin
                m1_awaddr_latch <= m1_axi_awaddr;
                m1_aw_pending <= 1'b1;
            end

            if (m1_axi_wvalid && m1_axi_wready) begin
                m1_wdata_latch <= m1_axi_wdata;
                m1_wstrb_latch <= m1_axi_wstrb;
                m1_w_pending <= 1'b1;
            end

            if (m1_aw_pending && m1_w_pending) begin
                ddr_memory[m1_awaddr_latch & ~(32'h7)] <= m1_wdata_latch;
                m1_aw_pending <= 1'b0;
                m1_w_pending <= 1'b0;
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
    // Helper Functions & Tasks
    // ==========================================
    function string trim_ws(string s);
        int i;
        int start;
        int finish;
        begin
            start = 0;
            finish = s.len() - 1;
            while (start <= finish && (s[start] == " " || s[start] == "\t" || s[start] == "\n" || s[start] == "\r")) begin
                start++;
            end
            while (finish >= start && (s[finish] == " " || s[finish] == "\t" || s[finish] == "\n" || s[finish] == "\r")) begin
                finish--;
            end
            if (finish < start) begin
                trim_ws = "";
            end else begin
                trim_ws = s.substr(start, finish);
            end
        end
    endfunction

    function string strip_ext(string s);
        int i;
        begin
            strip_ext = s;
            for (i = s.len() - 1; i >= 0; i--) begin
                if (s[i] == ".") begin
                    if (i > 0) strip_ext = s.substr(0, i - 1);
                    return strip_ext;
                end
                if (s[i] == "/") begin
                    return strip_ext;
                end
            end
            return strip_ext;
        end
    endfunction

    task parse_frame_list(input string list_str);
        int i;
        int start;
        string token;
        begin
            start = 0;
            for (i = 0; i <= list_str.len(); i++) begin
                if (i == list_str.len() || list_str[i] == ",") begin
                    if (i > start) begin
                        token = list_str.substr(start, i - 1);
                        token = trim_ws(token);
                        if (token.len() > 0 && frame_count < MAX_FRAMES) begin
                            frame_list[frame_count] = token;
                            frame_count++;
                        end
                    end
                    start = i + 1;
                end
            end
        end
    endtask

    task load_frames_from_file(input string path);
        int fd;
        string line;
        begin
            fd = $fopen(path, "r");
            if (!fd) begin
                $display("WARNING: Could not open frame_list_file: %s", path);
                return;
            end
            while (!$feof(fd) && frame_count < MAX_FRAMES) begin
                line = "";
                void'($fgets(line, fd));
                line = trim_ws(line);
                if (line.len() == 0) begin
                    continue;
                end
                if (line.len() > 0 && line.substr(0, 0) == "#") begin
                    continue;
                end
                frame_list[frame_count] = line;
                frame_count++;
            end
            $fclose(fd);
        end
    endtask

    task open_ref_file(
        input string primary,
        input string fallback,
        output int fd,
        output bit valid
    );
        begin
            fd = $fopen(primary, "r");
            if (!fd && fallback.len() > 0) begin
                fd = $fopen(fallback, "r");
            end
            valid = (fd != 0);
            if (!valid) begin
                $display("WARNING: Missing reference file: %s", primary);
            end
        end
    endtask

    task open_frame_files(input string stem);
        string golden_dir;
        string out_dir;
        begin
            golden_dir = $sformatf("%s/%s", golden_root, stem);
            out_dir = $sformatf("%s/%s", out_root, stem);
            void'($system($sformatf("mkdir -p %s", out_dir)));

            open_ref_file($sformatf("%s/mask_out.txt", golden_dir),
                          $sformatf("%s/mask_out.txt", data_root),
                          f_mask_ref, mask_ref_valid);
            open_ref_file($sformatf("%s/morph_out.txt", golden_dir),
                          $sformatf("%s/morph_out.txt", data_root),
                          f_morph_ref, morph_ref_valid);
            open_ref_file($sformatf("%s/ccl_pass1_golden.txt", golden_dir),
                          $sformatf("%s/ccl_pass1_golden.txt", data_root),
                          f_ccl_ref, ccl_ref_valid);
            open_ref_file($sformatf("%s/geom_bboxes_golden.txt", golden_dir),
                          $sformatf("%s/geom_bboxes_golden.txt", data_root),
                          f_geom_ref, geom_ref_valid);
            open_ref_file($sformatf("%s/template_matching_golden.txt", golden_dir),
                          $sformatf("%s/template_matching_golden.txt", data_root),
                          f_template_ref, template_ref_valid);

            f_mask_actual  = $fopen($sformatf("%s/actual_mask_out_e2e.txt", out_dir), "w");
            f_morph_actual = $fopen($sformatf("%s/actual_morph_out_e2e.txt", out_dir), "w");
            f_ccl_actual   = $fopen($sformatf("%s/actual_ccl_pass1_e2e.txt", out_dir), "w");
            f_ccl_pass2    = $fopen($sformatf("%s/actual_ccl_pass2_e2e.txt", out_dir), "w");
            f_ccl_resolved = $fopen($sformatf("%s/rtl_ccl_resolved_e2e.txt", out_dir), "w");
            f_ccl_stats_e2e = $fopen($sformatf("%s/actual_ccl_stats_e2e.txt", out_dir), "w");
            f_geom_filtered= $fopen($sformatf("%s/actual_geom_filtered_e2e.txt", out_dir), "w");
            f_rtl_roi      = $fopen($sformatf("%s/rtl_fetched_gray_e2e.hex", out_dir), "w");
            f_rtl_results  = $fopen($sformatf("%s/rtl_final_results_e2e.txt", out_dir), "w");
            f_template_actual = $fopen($sformatf("%s/actual_template_matching_e2e.txt", out_dir), "w");
            f_scores_full = $fopen($sformatf("%s/actual_scores_full_e2e.txt", out_dir), "w");
            f_roi_bin = $fopen($sformatf("%s/actual_roi_bin_0_e2e.txt", out_dir), "w");

            if (f_template_actual) begin
                $fdisplay(f_template_actual, "# roi_id xmin xmax ymin ymax best_class_id best_score");
            end
            if (f_scores_full) begin
                $fdisplay(f_scores_full, "# template_id score");
            end
            if (f_roi_bin) begin
                $fdisplay(f_roi_bin, "# bin_row_hex");
            end
        end
    endtask

    task close_frame_files();
        begin
            if (f_mask_ref) $fclose(f_mask_ref);
            if (f_morph_ref) $fclose(f_morph_ref);
            if (f_ccl_ref) $fclose(f_ccl_ref);
            if (f_geom_ref) $fclose(f_geom_ref);
            if (f_template_ref) $fclose(f_template_ref);

            if (f_mask_actual) $fclose(f_mask_actual);
            if (f_morph_actual) $fclose(f_morph_actual);
            if (f_ccl_actual) $fclose(f_ccl_actual);
            if (f_ccl_pass2) $fclose(f_ccl_pass2);
            if (f_ccl_resolved) $fclose(f_ccl_resolved);
            if (f_ccl_stats_e2e) $fclose(f_ccl_stats_e2e);
            if (f_geom_filtered) $fclose(f_geom_filtered);
            if (f_rtl_roi) $fclose(f_rtl_roi);
            if (f_rtl_results) $fclose(f_rtl_results);
            if (f_template_actual) $fclose(f_template_actual);
            if (f_scores_full) $fclose(f_scores_full);
            if (f_roi_bin) $fclose(f_roi_bin);

            f_mask_ref = 0;
            f_morph_ref = 0;
            f_ccl_ref = 0;
            f_geom_ref = 0;
            f_template_ref = 0;

            f_mask_actual = 0;
            f_morph_actual = 0;
            f_ccl_actual = 0;
            f_ccl_pass2 = 0;
            f_ccl_resolved = 0;
            f_ccl_stats_e2e = 0;
            f_geom_filtered = 0;
            f_rtl_roi = 0;
            f_rtl_results = 0;
            f_template_actual = 0;
            f_scores_full = 0;
            f_roi_bin = 0;
        end
    endtask

    task reset_frame_counters();
        begin
            mask_errors = 0;
            morph_errors = 0;
            ccl_errors = 0;
            geom_errors = 0;
            geom_detected_count = 0;
            roi_pixel_cnt = 0;
            axi_errors = 0;
            template_errors = 0;
            frame_best_score = 0;
            frame_best_id = 0;
            frame_best_valid = 1'b0;
            scores_dumped = 1'b0;
            roi_bin_dumped = 1'b0;
        end
    endtask

    task dump_ccl_stats_e2e();
        int lbl;
        begin
            if (!f_ccl_stats_e2e) begin
                return;
            end
            $fdisplay(f_ccl_stats_e2e, "# label area perimeter xmin xmax ymin ymax");
            for (lbl = 1; lbl < (1 << 16); lbl = lbl + 1) begin
                if (dut.u_backend_processing.u_ccl_stats.area_ram[lbl] != 0) begin
                    $fdisplay(f_ccl_stats_e2e, "%0d %0d %0d %0d %0d %0d %0d",
                              lbl,
                              dut.u_backend_processing.u_ccl_stats.area_ram[lbl],
                              dut.u_backend_processing.u_ccl_stats.perim_ram[lbl],
                              dut.u_backend_processing.u_ccl_stats.xmin_ram[lbl],
                              dut.u_backend_processing.u_ccl_stats.xmax_ram[lbl],
                              dut.u_backend_processing.u_ccl_stats.ymin_ram[lbl],
                              dut.u_backend_processing.u_ccl_stats.ymax_ram[lbl]);
                end
            end
        end
    endtask

    task load_expected_template();
        int rid;
        int xmin;
        int xmax;
        int ymin;
        int ymax;
        int bid;
        int bscore;
        string line;
        begin
            expected_best_id = -1;
            expected_best_score = 32'h7FFFFFFF;
            expected_no_sign = 1'b0;

            if (!template_ref_valid) begin
                expected_no_sign = 1'b0;
                return;
            end

            while (!$feof(f_template_ref)) begin
                line = "";
                void'($fgets(line, f_template_ref));
                if (line.len() == 0) begin
                    continue;
                end
                if (line.len() > 0 && line.substr(0, 0) == "#") begin
                    continue;
                end
                if ($sscanf(line, "%d %d %d %d %d %d %d", rid, xmin, xmax, ymin, ymax, bid, bscore) == 7) begin
                    expected_best_id = bid;
                    expected_best_score = bscore;
                    break;
                end
            end

            if (expected_best_score >= NO_SIGN_SCORE_THRESHOLD || expected_best_id < 0) begin
                expected_no_sign = 1'b1;
            end
        end
    endtask

    task resolve_image_hex(input string stem, output string hex_path);
        string cand;
        int fd;
        begin
            hex_path = "";

            cand = $sformatf("%s/%s/image_in.hex", golden_root, stem);
            fd = $fopen(cand, "r");
            if (fd) begin
                $fclose(fd);
                hex_path = cand;
                return;
            end

            cand = $sformatf("%s/%s/image_in2.hex", golden_root, stem);
            fd = $fopen(cand, "r");
            if (fd) begin
                $fclose(fd);
                hex_path = cand;
                return;
            end

            cand = $sformatf("%s/image_in.hex", data_root);
            fd = $fopen(cand, "r");
            if (fd) begin
                $fclose(fd);
                hex_path = cand;
                return;
            end

            $display("ERROR: Could not find image hex for %s", stem);
        end
    endtask

    task setup_and_enable();
        begin
            axi_lite_write(12'h014, 32'h8000_0000);
            axi_lite_write(12'h000, 32'h0000_0001);
        end
    endtask

    task soft_clear_between_frames();
        begin
            $display("[%0t] CLEAR: soft clear between frames (next frame=%s pass=%0d)", $time, current_frame_name, current_pass);
            axi_lite_write(12'h000, 32'h0000_0000);
            repeat (20) @(posedge clk);
            axi_lite_write(12'h000, 32'h0000_0001);
        end
    endtask

    task hard_reset_and_enable();
        begin
            $display("[%0t] CLEAR: hard reset between frames (next frame=%s pass=%0d)", $time, current_frame_name, current_pass);
            rst_n = 1'b0;
            repeat (20) @(posedge clk);
            rst_n = 1'b1;
            repeat (20) @(posedge clk);
            setup_and_enable();
        end
    endtask

    // ==========================================
    // Template Mapping (Descriptions)
    // ==========================================
    task load_template_desc();
        int fd_map;
        string line_str;
        int id_val;
        int dash_idx;
        string desc_str;
        begin
            for (int i = 0; i < 20; i++) begin
                template_desc[i] = "Unknown";
            end

            fd_map = $fopen($sformatf("%s/data/template_mapping.txt", PROJECT_A_DIR), "r");
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
        end
    endtask

    // ==========================================
    // Intermediate Signals Monitoring & Comparison
    // ==========================================
    bit morph_sync_done = 0;
    int morph_pixel_cnt = 0;
    int ccl_pixel_cnt = 0;
    logic [2:0] prev_backend_state;

    always @(posedge clk) begin
        if (!frame_active) begin
            morph_sync_done <= 1'b0;
        end
    end

    // Monitor Red Mask Output
    always @(posedge clk) begin
        if (frame_active && dut.u_red_mask.m_axis_tvalid && dut.u_morphology.s_axis_tready) begin
            logic ref_bit;
            int status;
            if (f_mask_actual) begin
                $fdisplay(f_mask_actual, "%b", dut.u_red_mask.m_axis_tdata);
            end
            if (mask_ref_valid) begin
                status = $fscanf(f_mask_ref, "%b", ref_bit);
                if (status != 1) begin
                    mask_ref_valid = 1'b0;
                end else if (dut.u_red_mask.m_axis_tdata !== ref_bit) begin
                    mask_errors++;
                end
            end
        end
    end

    // Monitor Morphology Output
    always @(posedge clk) begin
        if (frame_active && dut.u_morphology.m_axis_tvalid && dut.u_morphology.m_axis_tready) begin
            if (dut.u_morphology.m_axis_tuser) morph_sync_done = 1'b1;

            if (morph_sync_done) begin
                logic ref_bit;
                int status;
                morph_pixel_cnt++;
                if (f_morph_actual) begin
                    $fdisplay(f_morph_actual, "%b", dut.u_morphology.m_axis_tdata);
                end

                if (morph_ref_valid) begin
                    status = $fscanf(f_morph_ref, "%b", ref_bit);
                    if (status != 1) begin
                        morph_ref_valid = 1'b0;
                    end else if (dut.u_morphology.m_axis_tdata !== ref_bit) begin
                        morph_errors++;
                    end
                end
            end
        end
    end

    // Monitor CCL Output
    always @(posedge clk) begin
        if (frame_active && dut.u_backend_processing.u_ccl_engine.p1_axis_tvalid) begin
            int ref_label;
            int rtl_label;
            int status;
            rtl_label = dut.u_backend_processing.u_ccl_engine.p1_axis_tdata;
            ccl_pixel_cnt++;
            if (f_ccl_actual) begin
                $fdisplay(f_ccl_actual, "%d", rtl_label);
            end

            if (ccl_ref_valid) begin
                status = $fscanf(f_ccl_ref, "%d", ref_label);
                if (status != 1) begin
                    ccl_ref_valid = 1'b0;
                end else if (rtl_label !== ref_label) begin
                    ccl_errors++;
                end
            end
        end
    end

    // Monitor Pass2 label stream into stats collector
    always @(posedge clk) begin
        if (frame_active && dut.u_backend_processing.p2_stream_tvalid) begin
            if (f_ccl_pass2) begin
                $fdisplay(f_ccl_pass2, "%0d", dut.u_backend_processing.p2_stream_tdata);
            end
        end
    end

    // Monitor CCL Resolved Output
    always @(posedge clk) begin
        if (frame_active && dut.u_backend_processing.u_ccl_engine.u_resolver.parent_we) begin
            if (f_ccl_resolved) begin
                $fdisplay(f_ccl_resolved, "%0d [%0d]", dut.u_backend_processing.u_ccl_engine.u_resolver.parent_addr,
                          dut.u_backend_processing.u_ccl_engine.u_resolver.parent_wdata);
            end
        end
    end

    // Monitor Geometry Filter Output
    logic geom_pending;
    logic [15:0] geom_label_q;
    logic [15:0] geom_xmin_q;
    logic [15:0] geom_xmax_q;
    logic [15:0] geom_ymin_q;
    logic [15:0] geom_ymax_q;
    bit geom_done_logged;
    bit roi_fetch_logged;

    always @(posedge clk) begin
        if (!frame_active) begin
            geom_pending <= 1'b0;
            geom_done_logged <= 1'b0;
            roi_fetch_logged <= 1'b0;
        end else begin
            if (!geom_done_logged && dut.u_backend_processing.geo_done) begin
                $display("[%0t] GEO_DONE: bbox=[%0d,%0d,%0d,%0d] label=%0d",
                         $time,
                         dut.u_backend_processing.sts_bbox_xmin,
                         dut.u_backend_processing.sts_bbox_xmax,
                         dut.u_backend_processing.sts_bbox_ymin,
                         dut.u_backend_processing.sts_bbox_ymax,
                         dut.u_backend_processing.u_geometry_filter.obj_label);
                geom_done_logged <= 1'b1;
            end

            if (!roi_fetch_logged && dut.u_backend_processing.state == 3'b101) begin
                $display("[%0t] ROI_FETCH: bbox=[%0d,%0d,%0d,%0d] label=%0d frame_written=%0b addr=0x%08x row=%0d last_row=%0b",
                         $time,
                         dut.u_backend_processing.sts_bbox_xmin,
                         dut.u_backend_processing.sts_bbox_xmax,
                         dut.u_backend_processing.sts_bbox_ymin,
                         dut.u_backend_processing.sts_bbox_ymax,
                         dut.u_backend_processing.u_geometry_filter.obj_label,
                         dut.u_rgb_writer.frame_written,
                         dut.u_rgb_writer.addr_offset,
                         dut.u_rgb_writer.row_cnt,
                         dut.u_rgb_writer.last_row_seen);
                roi_fetch_logged <= 1'b1;
            end

            if (geom_pending) begin
                if (f_geom_filtered) begin
                    $fdisplay(f_geom_filtered, "%0d %0d %0d %0d %0d",
                              geom_label_q, geom_xmin_q, geom_xmax_q, geom_ymin_q, geom_ymax_q);
                end

                geom_detected_count++;

                if (geom_ref_valid) begin
                    int ref_label;
                    int ref_xmin;
                    int ref_xmax;
                    int ref_ymin;
                    int ref_ymax;
                    int status;
                    status = $fscanf(f_geom_ref, "%d %d %d %d %d", ref_label, ref_xmin, ref_xmax, ref_ymin, ref_ymax);
                    if (status != 5) begin
                        geom_ref_valid = 1'b0;
                    end else if (ref_label != geom_label_q || ref_xmin != geom_xmin_q || ref_xmax != geom_xmax_q ||
                                 ref_ymin != geom_ymin_q || ref_ymax != geom_ymax_q) begin
                        geom_errors++;
                    end
                end

                geom_pending <= 1'b0;
            end

            if (dut.u_backend_processing.u_geometry_filter.obj_valid) begin
                geom_label_q <= dut.u_backend_processing.u_geometry_filter.obj_label;
                geom_xmin_q  <= dut.u_backend_processing.u_geometry_filter.obj_xmin;
                geom_xmax_q  <= dut.u_backend_processing.u_geometry_filter.obj_xmax;
                geom_ymin_q  <= dut.u_backend_processing.u_geometry_filter.obj_ymin;
                geom_ymax_q  <= dut.u_backend_processing.u_geometry_filter.obj_ymax;
                geom_pending <= 1'b1;
            end
        end
    end

    // Monitor: Dump fetched ROI to Hex File
    always @(posedge clk) begin
        if (frame_active && dut.u_backend_processing.u_matcher.state == 3'b001) begin
            if (dut.u_backend_processing.u_matcher.s_axis_gray_tvalid) begin
                if (f_rtl_roi) begin
                    $fdisplay(f_rtl_roi, "%02x", dut.u_backend_processing.u_matcher.s_axis_gray_tdata);
                end
                roi_pixel_cnt++;
            end
        end
    end

    always @(posedge clk) begin
        if (frame_active && dut.u_backend_processing.u_matcher.match_done) begin
            frame_best_score <= dut.u_backend_processing.u_matcher.best_score;
            frame_best_id <= dut.u_backend_processing.u_matcher.best_class_id;
            frame_best_valid <= 1'b1;
            if (f_scores_full && !scores_dumped) begin
                for (int tid = 0; tid < TEMPLATE_COUNT; tid++) begin
                    $fdisplay(f_scores_full, "%0d %0d", tid, dut.u_backend_processing.u_matcher.debug_scores[tid]);
                end
                scores_dumped <= 1'b1;
            end
            if (f_roi_bin && !roi_bin_dumped) begin
                for (int r = 0; r < 32; r++) begin
                    $fdisplay(f_roi_bin, "%08x", dut.u_backend_processing.u_matcher.bin_roi[r]);
                end
                roi_bin_dumped <= 1'b1;
            end
        end
    end

    // ==========================================
    // AXI Protocol Checks
    // ==========================================
    logic m0_aw_hold;
    logic m0_w_hold;
    logic m1_aw_hold;
    logic m1_w_hold;
    logic m1_ar_hold;
    logic m1_r_hold;

    logic [AXI_FULL_ADDR_W-1:0] m0_awaddr_hold;
    logic [AXI_FULL_DATA_W-1:0] m0_wdata_hold;
    logic [7:0]                 m0_wstrb_hold;
    logic [AXI_FULL_ADDR_W-1:0] m1_awaddr_hold;
    logic [AXI_FULL_DATA_W-1:0] m1_wdata_hold;
    logic [7:0]                 m1_wstrb_hold;
    logic [AXI_FULL_ADDR_W-1:0] m1_araddr_hold;
    logic [AXI_FULL_DATA_W-1:0] m1_rdata_hold;
    logic                       prev_frame_written;
    logic                       sof_post_pending;
    string                      sof_frame_name_q;
    int                         sof_pass_q;

    // Stream protocol check
    always @(posedge clk) begin
        if (dut.frame_active && s_axis_tvalid && s_axis_tready && s_axis_tuser) begin
            axi_errors++;
            $display("[%0t] AXIS ERROR: SOF asserted while frame_active", $time);
        end
    end

    // Debug: SOF acceptance and frame-written timing
    always @(posedge clk) begin
        if (s_axis_tvalid && s_axis_tuser && !s_axis_tready) begin
            $display("[%0t] SOF_STALL: frame=%s pass=%0d", $time, current_frame_name, current_pass);
        end
        if (s_axis_tvalid && s_axis_tready && s_axis_tuser) begin
            $display("[%0t] SOF_ACCEPTED: frame=%s pass=%0d fw=%0b addr=0x%08x row=%0d",
                     $time, current_frame_name, current_pass,
                     dut.u_rgb_writer.frame_written,
                     dut.u_rgb_writer.addr_offset,
                     dut.u_rgb_writer.row_cnt);
            sof_post_pending <= 1'b1;
            sof_frame_name_q = current_frame_name;
            sof_pass_q <= current_pass;
        end
        if (sof_post_pending) begin
            $display("[%0t] SOF_POST: frame=%s pass=%0d fw=%0b addr=0x%08x row=%0d last_row=%0b pix_idx=%0b state=%0d",
                     $time, sof_frame_name_q, sof_pass_q,
                     dut.u_rgb_writer.frame_written,
                     dut.u_rgb_writer.addr_offset,
                     dut.u_rgb_writer.row_cnt,
                     dut.u_rgb_writer.last_row_seen,
                     dut.u_rgb_writer.pixel_idx,
                     dut.u_rgb_writer.state);
            sof_post_pending <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_frame_written <= 1'b0;
            sof_post_pending <= 1'b0;
            sof_frame_name_q = "";
        end else begin
            if (!prev_frame_written && dut.u_rgb_writer.frame_written) begin
                $display("[%0t] FRAME_WRITTEN: asserted (frame=%s pass=%0d addr=0x%08x row=%0d last_row=%0b pix_idx=%0b)",
                         $time, current_frame_name, current_pass,
                         dut.u_rgb_writer.addr_offset,
                         dut.u_rgb_writer.row_cnt,
                         dut.u_rgb_writer.last_row_seen,
                         dut.u_rgb_writer.pixel_idx);
            end
            prev_frame_written <= dut.u_rgb_writer.frame_written;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m0_aw_hold <= 1'b0;
            m0_w_hold  <= 1'b0;
            m1_aw_hold <= 1'b0;
            m1_w_hold  <= 1'b0;
            m1_ar_hold <= 1'b0;
            m1_r_hold  <= 1'b0;
        end else if (!frame_active) begin
            m0_aw_hold <= 1'b0;
            m0_w_hold  <= 1'b0;
            m1_aw_hold <= 1'b0;
            m1_w_hold  <= 1'b0;
            m1_ar_hold <= 1'b0;
            m1_r_hold  <= 1'b0;
        end else begin
            if (m0_axi_awvalid && !m0_axi_awready) begin
                if (!m0_aw_hold) begin
                    m0_aw_hold <= 1'b1;
                    m0_awaddr_hold <= m0_axi_awaddr;
                end else if (m0_axi_awaddr != m0_awaddr_hold) begin
                    axi_errors++;
                end
            end else begin
                m0_aw_hold <= 1'b0;
            end

            if (m0_axi_wvalid && !m0_axi_wready) begin
                if (!m0_w_hold) begin
                    m0_w_hold <= 1'b1;
                    m0_wdata_hold <= m0_axi_wdata;
                    m0_wstrb_hold <= m0_axi_wstrb;
                end else if (m0_axi_wdata != m0_wdata_hold || m0_axi_wstrb != m0_wstrb_hold) begin
                    axi_errors++;
                end
            end else begin
                m0_w_hold <= 1'b0;
            end

            if (m1_axi_awvalid && !m1_axi_awready) begin
                if (!m1_aw_hold) begin
                    m1_aw_hold <= 1'b1;
                    m1_awaddr_hold <= m1_axi_awaddr;
                end else if (m1_axi_awaddr != m1_awaddr_hold) begin
                    axi_errors++;
                end
            end else begin
                m1_aw_hold <= 1'b0;
            end

            if (m1_axi_wvalid && !m1_axi_wready) begin
                if (!m1_w_hold) begin
                    m1_w_hold <= 1'b1;
                    m1_wdata_hold <= m1_axi_wdata;
                    m1_wstrb_hold <= m1_axi_wstrb;
                end else if (m1_axi_wdata != m1_wdata_hold || m1_axi_wstrb != m1_wstrb_hold) begin
                    axi_errors++;
                end
            end else begin
                m1_w_hold <= 1'b0;
            end

            if (m1_axi_arvalid && !m1_axi_arready) begin
                if (!m1_ar_hold) begin
                    m1_ar_hold <= 1'b1;
                    m1_araddr_hold <= m1_axi_araddr;
                end else if (m1_axi_araddr != m1_araddr_hold) begin
                    axi_errors++;
                end
            end else begin
                m1_ar_hold <= 1'b0;
            end

            if (m1_axi_rvalid && !m1_axi_rready) begin
                if (!m1_r_hold) begin
                    m1_r_hold <= 1'b1;
                    m1_rdata_hold <= m1_axi_rdata;
                end else if (m1_axi_rdata != m1_rdata_hold) begin
                    axi_errors++;
                end
            end else begin
                m1_r_hold <= 1'b0;
            end
        end
    end

    // ==========================================
    // Main Test Sequence
    // ==========================================
    logic [31:0] read_val;
    logic [7:0] class_id;
    logic [15:0] bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax;
    int prev_last_valid_class_id;

    initial begin
        int tmp;
        string frames_arg;
        string list_file;
        string image_hex_path;
        int pass_count;
        int current_id_sampled;
        int last_valid_id_sampled;
        bit do_clear;

        // Open summary; write header after mode/frame args are resolved.
        f_summary = $fopen($sformatf("%s/results/e2e_summary.txt", PROJECT_A_DIR), "w");

        frame_active = 1'b0;
        frame_count = 0;
        mode = "single";
        single_frame = "14.png";
        golden_root = $sformatf("%s/results/by_image", PROJECT_A_DIR);
        out_root = $sformatf("%s/results/by_image", PROJECT_A_DIR);
        data_root = $sformatf("%s/data", PROJECT_A_DIR);
        check_clear = 1'b0;
        hard_reset_between_frames = 1'b0;
        force_clear_between_frames = 1'b0;
        last_valid_class_id = 0;

        if ($value$plusargs("mode=%s", mode)) begin
            mode = trim_ws(mode);
        end
        if ($value$plusargs("single=%s", single_frame)) begin
            single_frame = trim_ws(single_frame);
        end
        if ($value$plusargs("frames=%s", frames_arg)) begin
            parse_frame_list(frames_arg);
        end
        if ($value$plusargs("frame_list_file=%s", list_file)) begin
            load_frames_from_file(list_file);
        end
        if ($value$plusargs("golden_root=%s", golden_root)) begin
            golden_root = trim_ws(golden_root);
        end
        if ($value$plusargs("out_root=%s", out_root)) begin
            out_root = trim_ws(out_root);
        end
        if ($value$plusargs("data_root=%s", data_root)) begin
            data_root = trim_ws(data_root);
        end
        if ($value$plusargs("check_clear=%d", tmp)) begin
            check_clear = (tmp != 0);
        end
        if ($value$plusargs("hard_reset_between_frames=%d", tmp)) begin
            hard_reset_between_frames = (tmp != 0);
        end
        if ($value$plusargs("clear_between_frames=%d", tmp)) begin
            force_clear_between_frames = (tmp != 0);
        end

        if (frame_count == 0) begin
            if (mode == "single") begin
                frame_list[0] = single_frame;
                frame_count = 1;
            end else begin
                frame_list[0] = "11.png";
                frame_list[1] = "14.png";
                frame_list[2] = "22.png";
                frame_count = 3;
                check_clear = 1'b1;
            end
        end

        if (f_summary) begin
            $fdisplay(f_summary, "E2E SUMMARY");
            $fdisplay(f_summary, "mode=%s", mode);
            $fdisplay(f_summary, "frames=%0d", frame_count);
            $fdisplay(f_summary, "golden_root=%s", golden_root);
            $fdisplay(f_summary, "out_root=%s", out_root);
            $fdisplay(f_summary, "");
            $fdisplay(f_summary, "%-4s %-16s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-12s %-16s %-12s",
                "Pass", "Frame", "Mask", "Morph", "CCL", "Geom", "AXI", "Tmpl", "Total", "Detected", "Template", "Detection", "Result");
        end

        load_template_desc();

        prev_backend_state = 3'b000;

        if (prev_backend_state !== dut.u_backend_processing.state) begin
            $display("[FSM] %3b -> %3b", prev_backend_state, dut.u_backend_processing.state);
            prev_backend_state = dut.u_backend_processing.state;
        end

        $display("[%0t] System Asserting Reset...", $time);
        rst_n = 0;
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 10);

        setup_and_enable();

        pass_count = check_clear ? 2 : 1;
        for (current_pass = 0; current_pass < pass_count; current_pass++) begin
            do_clear = (current_pass == 1) ? 1'b1 : force_clear_between_frames;
            if (current_pass == 1) begin
                last_valid_class_id = 0;
                if (hard_reset_between_frames) begin
                    hard_reset_and_enable();
                end
            end

            for (current_frame = 0; current_frame < frame_count; current_frame++) begin
                current_frame_name = frame_list[current_frame];
                current_stem = strip_ext(current_frame_name);
                reset_frame_counters();
                morph_pixel_cnt = 0;
                ccl_pixel_cnt = 0;

                open_frame_files(current_stem);
                load_expected_template();
                resolve_image_hex(current_stem, image_hex_path);
                
                if (image_hex_path.len() == 0) begin
                    $display("ERROR: Missing image hex for %s", current_stem);
                    $finish;
                end

                if (current_frame > 0 && do_clear) begin
                    if (hard_reset_between_frames) begin
                        hard_reset_and_enable();
                    end else begin
                        soft_clear_between_frames();
                    end
                end

                frame_active = 1'b1;
                $display("[%0t] Streaming frame %0d: %s (pass %0d)", $time, current_frame, current_frame_name, current_pass);
                stream_rgb_file(image_hex_path);

                // Wait for IRQ with timeout
                begin : wait_irq
                    int c;
                    for (c = 0; c < 50000000; c++) begin
                        @(posedge clk);
                        if (irq) disable wait_irq;
                    end
                    $display("ERROR: IRQ timeout for frame %s", current_frame_name);
                    $finish;
                end

                axi_lite_read(12'h004, read_val);
                class_id = (read_val >> 8) & 8'hFF;

                axi_lite_read(12'h008, read_val);
                bbox_xmin = read_val & 16'hFFFF;
                bbox_xmax = (read_val >> 16) & 16'hFFFF;

                axi_lite_read(12'h00C, read_val);
                bbox_ymin = read_val & 16'hFFFF;
                bbox_ymax = (read_val >> 16) & 16'hFFFF;

                prev_last_valid_class_id = dut.last_valid_id_reg;
                current_id_sampled = current_id;
                // Allow one cycle for current_id update after sts_done_flag.
                if (geom_detected_count == 0 && current_id_sampled != 255) begin
                    @(posedge clk);
                    current_id_sampled = current_id;
                end
                last_valid_id_sampled = dut.last_valid_id_reg;
                last_valid_class_id = last_valid_id_sampled;

                $display("[%0t] ID_CHECK: frame=%s pass=%0d geom_detected=%0d current_id=%0d last_valid_id=%0d csr_class=%0d",
                         $time, current_frame_name, current_pass,
                         geom_detected_count, current_id_sampled, last_valid_id_sampled, class_id);

                if (geom_detected_count == 0) begin
                    $display("[DETECTION] No sign candidate (no bbox). current_id=%0d last_valid_id=%0d",
                             current_id_sampled, last_valid_id_sampled);
                end else begin
                    $display("[DETECTION] Sign candidate exists. bbox=[%0d,%0d,%0d,%0d] current_id=%0d",
                             bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax, current_id_sampled);
                end

                if (geom_detected_count == 0) begin
                    // No bbox detected: only check that current_id==255 (no sign) and last_valid_id is unchanged
                    if (current_id_sampled != 255) begin
                        template_errors++;
                    end
                    if (last_valid_id_sampled != prev_last_valid_class_id) begin
                        template_errors++;
                    end
                end else begin
                    // Bbox detected: perform normal correctness checks
                    if (expected_no_sign) begin
                        if (last_valid_class_id != class_id && !(last_valid_class_id == 0 && class_id == 0)) begin
                            template_errors++;
                        end
                    end else begin
                        if (expected_best_id >= 0 && class_id != expected_best_id) begin
                            template_errors++;
                        end else begin
                            last_valid_class_id = class_id;
                        end
                    end
                end

                if (f_rtl_results) begin
                    $fdisplay(f_rtl_results, "FRAME: %s", current_frame_name);
                    $fdisplay(f_rtl_results, "CLASS_ID: %0d (%s)", class_id, template_desc[class_id]);
                    $fdisplay(f_rtl_results, "BBOX_X: [%0d, %0d]", bbox_xmin, bbox_xmax);
                    $fdisplay(f_rtl_results, "BBOX_Y: [%0d, %0d]", bbox_ymin, bbox_ymax);
                    $fdisplay(f_rtl_results, "MASK_ERRORS: %0d", mask_errors);
                    $fdisplay(f_rtl_results, "MORPH_ERRORS: %0d", morph_errors);
                    $fdisplay(f_rtl_results, "CCL_ERRORS: %0d", ccl_errors);
                    $fdisplay(f_rtl_results, "GEOM_ERRORS: %0d", geom_errors);
                    $fdisplay(f_rtl_results, "AXI_ERRORS: %0d", axi_errors);
                    $fdisplay(f_rtl_results, "TEMPLATE_ERRORS: %0d", template_errors);
                    $fdisplay(f_rtl_results, "ROI_PIXELS_FETCHED: %0d", roi_pixel_cnt);
                    $fdisplay(f_rtl_results, "GEOM_DETECTED: %0d", geom_detected_count);
                    $fdisplay(f_rtl_results, "CURRENT_ID: %0d", current_id_sampled);
                    $fdisplay(f_rtl_results, "LAST_VALID_ID: %0d", last_valid_id_sampled);
                end

                if (f_template_actual && frame_best_valid) begin
                    $fdisplay(f_template_actual, "0 %0d %0d %0d %0d %0d %0d",
                              bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax,
                              frame_best_id, frame_best_score);
                end

                dump_ccl_stats_e2e();

                pass_mask_errors[current_pass][current_frame] = mask_errors;
                pass_morph_errors[current_pass][current_frame] = morph_errors;
                pass_ccl_errors[current_pass][current_frame] = ccl_errors;
                pass_geom_errors[current_pass][current_frame] = geom_errors;
                pass_axi_errors[current_pass][current_frame] = axi_errors;
                pass_tmpl_errors[current_pass][current_frame] = template_errors;
                pass_total_errors[current_pass][current_frame] = mask_errors + morph_errors + ccl_errors + geom_errors + axi_errors + template_errors;

                if (f_summary) begin
                    string detected_str, template_str, detection_str, result_str;
                    detected_str = (geom_detected_count > 0) ? "YES" : "NO";
                    if (current_id_sampled == 255) begin
                        template_str = "NO";
                    end else begin
                        template_str = template_desc[current_id_sampled];
                    end
                    if (geom_detected_count == 0) begin
                        detection_str = (current_id_sampled == 255) ? "NO_BBOX_NO_SIGN" : "NO_BBOX_WRONG";
                    end else if (expected_no_sign) begin
                        detection_str = (class_id == 0) ? "CORRECT" : "WRONG";
                    end else begin
                        detection_str = (class_id == expected_best_id) ? "CORRECT" : "WRONG";
                    end
                    result_str = (pass_total_errors[current_pass][current_frame] == 0) ? "PASS" : "FAIL";
                    $fdisplay(f_summary, "%-4d %-16s %-8d %-8d %-8d %-8d %-8d %-8d %-8d %-8s %-12s %-16s %-12s",
                        current_pass, current_frame_name,
                        mask_errors, morph_errors, ccl_errors, geom_errors, axi_errors, template_errors,
                        pass_total_errors[current_pass][current_frame],
                        detected_str, template_str, detection_str, result_str);
                end

                frame_active = 1'b0;
                axi_lite_write(12'h000, 32'h0000_0003);
                close_frame_files();
            end
        end

        if (check_clear && f_summary) begin
            $fdisplay(f_summary, "");
            $fdisplay(f_summary, "CLEARING_ANALYSIS");
            for (frame_idx = 0; frame_idx < frame_count; frame_idx = frame_idx + 1) begin
                int err_no_clear = pass_total_errors[0][frame_idx];
                int err_with_clear = pass_total_errors[1][frame_idx];
                if (err_no_clear > 0 && err_with_clear == 0) begin
                    $fdisplay(f_summary, "frame=%s CLEAR_REQUIRED", frame_list[frame_idx]);
                end else begin
                    $fdisplay(f_summary, "frame=%s CLEAR_NOT_REQUIRED", frame_list[frame_idx]);
                end
            end
        end

        if (f_summary) begin
            $fclose(f_summary);
        end

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
    task automatic stream_rgb_file(string filename);
        int idx;

        idx = 0;
        $display("[%0t] Loading image into memory array: %s", $time, filename);
        $readmemh(filename, image_array);
        $display("[%0t] Loading complete. Streaming to DUT...", $time);

        for (int y = 0; y < IMG_H; y++) begin
            for (int x = 0; x < IMG_W; x++) begin
                s_axis_tvalid <= 1'b1;
                s_axis_tdata  <= image_array[idx][23:0];
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