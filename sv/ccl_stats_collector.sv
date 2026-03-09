module ccl_stats_collector #(
    parameter int LABEL_W = 9,
    parameter int COORD_W = 12,
    parameter int MAX_COORD = 4095
)(
    input  logic               clk,
    input  logic               rst_n,

    // Stream from DDR (Labels)
    input  logic [LABEL_W-1:0] s_axis_label,
    input  logic               s_axis_tvalid,
    input  logic               s_axis_tuser,
    input  logic               s_axis_tlast,

    // Interface to Parent RAM
    output logic [LABEL_W-1:0] parent_addr,
    input  logic [LABEL_W-1:0] parent_rdata,

    output logic               stats_done
);

    // ==========================================
    // Initialization FSM
    // ==========================================
    typedef enum logic [1:0] {
        ST_INIT,
        ST_PROCESS,
        ST_DONE
    } state_t;
    
    state_t state, next_state;
    logic [LABEL_W-1:0] init_addr;
    logic init_en;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_INIT;
            init_addr <= '0;
        end else begin
            case (state)
                ST_INIT: begin
                    if (init_addr == {LABEL_W{1'b1}}) begin
                        state <= ST_PROCESS;
                    end else begin
                        init_addr <= init_addr + 1;
                    end
                end
                ST_PROCESS: begin
                    if (s_axis_tvalid && s_axis_tuser) begin
                        state     <= ST_INIT;
                        init_addr <= '0;
                    end else if (s_axis_tvalid && s_axis_tlast && y_cnt == 1080 - 1) begin // Assuming 1080p
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (s_axis_tvalid && s_axis_tuser) begin
                        state     <= ST_INIT;
                        init_addr <= '0;
                    end
                end
            endcase
        end
    end

    assign init_en    = (state == ST_INIT);
    assign stats_done = (state == ST_DONE);

    // ==========================================
    // Coordinates Tracking
    // ==========================================
    logic [COORD_W-1:0] x_cnt, y_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_cnt <= '0;
            y_cnt <= '0;
        end else if (state == ST_INIT) begin
            x_cnt <= '0;
            y_cnt <= '0;
        end else if (s_axis_tvalid) begin
            if (s_axis_tlast) begin
                x_cnt <= '0;
                y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end
    end

    // ==========================================
    // Pipeline Registers
    // ==========================================
    
    // Stage 1: Request Root
    logic               stg1_valid;
    logic [COORD_W-1:0] stg1_x, stg1_y;
    logic [LABEL_W-1:0] stg1_label;

    // Stage 2: Request Stats
    logic               stg2_valid;
    logic [COORD_W-1:0] stg2_x, stg2_y;
    logic [LABEL_W-1:0] stg2_root;

    // Stage 3: Compute & Writeback
    logic               stg3_valid;
    logic [COORD_W-1:0] stg3_x, stg3_y;
    logic [LABEL_W-1:0] stg3_root;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg1_valid <= 1'b0;
            stg2_valid <= 1'b0;
            stg3_valid <= 1'b0;
        end else begin
            stg1_valid <= s_axis_tvalid && (s_axis_label != 0) && (state == ST_PROCESS);
            stg2_valid <= stg1_valid;
            stg3_valid <= stg2_valid;
            
            if (s_axis_tvalid) begin
                stg1_x <= x_cnt;
                stg1_y <= y_cnt;
                stg1_label <= s_axis_label;
            end
            
            if (stg1_valid) begin
                stg2_x <= stg1_x;
                stg2_y <= stg1_y;
            end
            
            if (stg2_valid) begin
                stg3_x <= stg2_x;
                stg3_y <= stg2_y;
                stg3_root <= stg2_root;
            end
        end
    end

    // ==========================================
    // Stage 1: Parent RAM Read
    // ==========================================
    assign parent_addr = s_axis_label;
    assign stg2_root   = parent_rdata;

    // ==========================================
    // RAMs Definition (Area, BBox)
    // ==========================================
    logic [31:0] area_ram [0:(1<<LABEL_W)-1];
    logic [15:0] xmin_ram [0:(1<<LABEL_W)-1];
    logic [15:0] xmax_ram [0:(1<<LABEL_W)-1];
    logic [15:0] ymin_ram [0:(1<<LABEL_W)-1];
    logic [15:0] ymax_ram [0:(1<<LABEL_W)-1];

    logic [LABEL_W-1:0] ram_addr_rd, ram_addr_wr;
    logic               ram_we;
    
    logic [31:0] area_rd_data, area_wr_data;
    logic [15:0] xmin_rd_data, xmin_wr_data;
    logic [15:0] xmax_rd_data, xmax_wr_data;
    logic [15:0] ymin_rd_data, ymin_wr_data;
    logic [15:0] ymax_rd_data, ymax_wr_data;

    assign ram_addr_rd = stg2_root;
    assign ram_addr_wr = init_en ? init_addr : stg3_root;
    assign ram_we      = init_en | stg3_valid;

    always_ff @(posedge clk) begin
        if (ram_we) begin
            area_ram[ram_addr_wr] <= init_en ? 32'd0 : area_wr_data;
            xmin_ram[ram_addr_wr] <= init_en ? MAX_COORD[15:0] : xmin_wr_data;
            xmax_ram[ram_addr_wr] <= init_en ? 16'd0 : xmax_wr_data;
            ymin_ram[ram_addr_wr] <= init_en ? MAX_COORD[15:0] : ymin_wr_data;
            ymax_ram[ram_addr_wr] <= init_en ? 16'd0 : ymax_wr_data;
        end
        area_rd_data <= area_ram[ram_addr_rd];
        xmin_rd_data <= xmin_ram[ram_addr_rd];
        xmax_rd_data <= xmax_ram[ram_addr_rd];
        ymin_rd_data <= ymin_ram[ram_addr_rd];
        ymax_rd_data <= ymax_ram[ram_addr_rd];
    end

    // ==========================================
    // Stage 3: Forwarding & Compute
    // ==========================================
    logic fwd_match;
    assign fwd_match = stg3_valid && (stg3_root == stg2_root);

    logic [31:0] fwd_area;
    logic [15:0] fwd_xmin, fwd_xmax, fwd_ymin, fwd_ymax;

    always_comb begin
        fwd_area = fwd_match ? area_wr_data : area_rd_data;
        fwd_xmin = fwd_match ? xmin_wr_data : xmin_rd_data;
        fwd_xmax = fwd_match ? xmax_wr_data : xmax_rd_data;
        fwd_ymin = fwd_match ? ymin_wr_data : ymin_rd_data;
        fwd_ymax = fwd_match ? ymax_wr_data : ymax_rd_data;

        area_wr_data = fwd_area + 1;
        
        xmin_wr_data = (stg3_x < fwd_xmin) ? stg3_x : fwd_xmin;
        xmax_wr_data = (stg3_x > fwd_xmax) ? stg3_x : fwd_xmax;
        ymin_wr_data = (stg3_y < fwd_ymin) ? stg3_y : fwd_ymin;
        ymax_wr_data = (stg3_y > fwd_ymax) ? stg3_y : fwd_ymax;
    end

endmodule