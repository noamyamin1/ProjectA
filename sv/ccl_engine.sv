module ccl_engine #(
    parameter int IMG_WIDTH  = 1920,
    parameter int IMG_HEIGHT = 1080,
    parameter int LABEL_W    = 9
)(
    input  logic               clk,
    input  logic               rst_n,

    // Input Stream (from Morphology)
    input  logic               s_axis_tdata,
    input  logic               s_axis_tvalid,
    input  logic               s_axis_tuser,
    input  logic               s_axis_tlast,

    // Control & Status
    output logic               pass1_done,
    output logic               resolver_done,

    // Interface for Pass 2 (Read-only access to Parent RAM)
    input  logic [LABEL_W-1:0] p2_parent_addr,
    output logic [LABEL_W-1:0] p2_parent_rdata
);

    // ==========================================
    // Internal Signals
    // ==========================================
    logic [LABEL_W-1:0] pass1_tdata;
    logic               pass1_tvalid;
    logic               pass1_tuser;
    logic               pass1_tlast;

    logic               pass1_parent_we;
    logic [LABEL_W-1:0] pass1_parent_addr;
    logic [LABEL_W-1:0] pass1_parent_wdata;

    logic               res_start;
    logic [LABEL_W-1:0] max_label_reg;
    
    logic               res_parent_we;
    logic [LABEL_W-1:0] res_parent_addr;
    logic [LABEL_W-1:0] res_parent_wdata;

    // RAM Ports
    logic               ram_we;
    logic [LABEL_W-1:0] ram_addr;
    logic [LABEL_W-1:0] ram_wdata;
    logic [LABEL_W-1:0] ram_rdata;

    // Row counter for End-Of-Frame detection
    logic [11:0] row_cnt;

    // FSM States
    typedef enum logic [1:0] {
        ST_PASS1   = 2'b00,
        ST_RESOLVE = 2'b01,
        ST_PASS2   = 2'b10
    } ccl_state_e;
    
    ccl_state_e curr_state;

    // ==========================================
    // Parent RAM (Inferred BRAM)
    // ==========================================
    logic [LABEL_W-1:0] parent_ram [0:(1<<LABEL_W)-1];

    always_ff @(posedge clk) begin
        if (ram_we) begin
            parent_ram[ram_addr] <= ram_wdata;
        end
        ram_rdata <= parent_ram[ram_addr];
    end

    // ==========================================
    // Pass 1 Labeler Instantiation
    // ==========================================
    ccl_pass1_labeler #(
        .IMG_WIDTH(IMG_WIDTH),
        .LABEL_W(LABEL_W)
    ) u_pass1 (
        .clk          (clk),
        .rst_n        (rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tuser (s_axis_tuser),
        .s_axis_tlast (s_axis_tlast),
        .m_axis_tdata (pass1_tdata),
        .m_axis_tvalid(pass1_tvalid),
        .m_axis_tuser (pass1_tuser),
        .m_axis_tlast (pass1_tlast),
        .parent_we    (pass1_parent_we),
        .parent_addr  (pass1_parent_addr),
        .parent_wdata (pass1_parent_wdata)
    );

    // ==========================================
    // UF Resolver Instantiation
    // ==========================================
    ccl_uf_resolver #(
        .LABEL_W(LABEL_W)
    ) u_resolver (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (res_start),
        .max_label   (max_label_reg),
        .parent_addr (res_parent_addr),
        .parent_we   (res_parent_we),
        .parent_wdata(res_parent_wdata),
        .parent_rdata(ram_rdata),
        .done        (resolver_done)
    );

    // ==========================================
    // State Machine & Max Label Tracker
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_label_reg <= '0;
            curr_state    <= ST_PASS1;
            res_start     <= 1'b0;
            pass1_done    <= 1'b0;
            row_cnt       <= '0;
        end else begin
            res_start <= 1'b0;
            
            case (curr_state)
                ST_PASS1: begin
                    if (s_axis_tvalid && s_axis_tuser) begin
                        max_label_reg <= '0; 
                        row_cnt       <= '0;
                    end
                    
                    if (pass1_tvalid && pass1_tdata > max_label_reg) begin
                        max_label_reg <= pass1_tdata;
                    end
                    
                    if (pass1_tvalid && pass1_tlast) begin
                        row_cnt <= row_cnt + 1;
                        if (row_cnt == IMG_HEIGHT - 1) begin
                            curr_state <= ST_RESOLVE;
                            res_start  <= 1'b1;
                            pass1_done <= 1'b1;
                        end
                    end
                end
                
                ST_RESOLVE: begin
                    if (resolver_done) begin
                        curr_state <= ST_PASS2;
                    end
                end
                
                ST_PASS2: begin
                    if (s_axis_tvalid && s_axis_tuser) begin
                        curr_state <= ST_PASS1;
                        pass1_done <= 1'b0;
                        row_cnt    <= '0;
                    end
                end
            endcase
        end
    end

    // ==========================================
    // RAM Multiplexer
    // ==========================================
    always_comb begin
        ram_we    = 1'b0;
        ram_addr  = '0;
        ram_wdata = '0;
        
        p2_parent_rdata = ram_rdata;

        case (curr_state)
            ST_PASS1: begin
                ram_we    = pass1_parent_we;
                ram_addr  = pass1_parent_addr;
                ram_wdata = pass1_parent_wdata;
            end
            ST_RESOLVE: begin
                ram_we    = res_parent_we;
                ram_addr  = res_parent_addr;
                ram_wdata = res_parent_wdata;
            end
            ST_PASS2: begin
                ram_we    = 1'b0;
                ram_addr  = p2_parent_addr;
                ram_wdata = '0;
            end
            default: ;
        endcase
    end

endmodule