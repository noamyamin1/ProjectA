module ccl_uf_resolver #(
    parameter int LABEL_W = 9
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start,
    input  logic [LABEL_W-1:0] max_label,

    output logic [LABEL_W-1:0] parent_addr,
    output logic               parent_we,
    output logic [LABEL_W-1:0] parent_wdata,
    input  logic [LABEL_W-1:0] parent_rdata,

    output logic               done
);

    // ==========================================
    // FSM States
    // ==========================================
    typedef enum logic [2:0] {
        ST_IDLE       = 3'b000,
        ST_REQ_RD     = 3'b001,
        ST_WAIT_RD    = 3'b010,
        ST_CHECK_ROOT = 3'b011,
        ST_WRITE_ROOT = 3'b100,
        ST_DONE       = 3'b101
    } state_t;

    state_t state, next_state;

    // ==========================================
    // Internal Registers
    // ==========================================
    logic [LABEL_W-1:0] curr_label;
    logic [LABEL_W-1:0] ptr_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end

    // ==========================================
    // FSM Next State & Datapath Logic
    // ==========================================
    always_comb begin
        next_state = state;
        
        case (state)
            ST_IDLE: begin
                if (start && max_label > 1) begin
                    next_state = ST_REQ_RD;
                end else if (start) begin
                    next_state = ST_DONE;
                end
            end
            
            ST_REQ_RD: begin
                next_state = ST_WAIT_RD;
            end
            
            ST_WAIT_RD: begin
                next_state = ST_CHECK_ROOT;
            end
            
            ST_CHECK_ROOT: begin
                if (parent_rdata == ptr_reg) begin
                    next_state = ST_WRITE_ROOT;
                end else begin
                    next_state = ST_REQ_RD;
                end
            end
            
            ST_WRITE_ROOT: begin
                if (curr_label == max_label - 1) begin
                    next_state = ST_DONE;
                end else begin
                    next_state = ST_REQ_RD;
                end
            end
            
            ST_DONE: begin
                if (!start) begin
                    next_state = ST_IDLE;
                end
            end
            
            default: next_state = ST_IDLE;
        endcase
    end

    // ==========================================
    // Sequential Updates
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            curr_label <= '0;
            ptr_reg    <= '0;
        end else begin
            case (state)
                ST_IDLE: begin
                    curr_label <= 9'd1;
                    ptr_reg    <= 9'd1;
                end
                
                ST_CHECK_ROOT: begin
                    if (parent_rdata != ptr_reg) begin
                        ptr_reg <= parent_rdata;
                    end
                end
                
                ST_WRITE_ROOT: begin
                    curr_label <= curr_label + 1;
                    ptr_reg    <= curr_label + 1;
                end
                
                default: ;
            endcase
        end
    end

    // ==========================================
    // Output Assignments
    // ==========================================
    assign parent_we = (state == ST_WRITE_ROOT);
    
    always_comb begin
        if (state == ST_WRITE_ROOT) begin
            parent_addr = curr_label;
        end else begin
            parent_addr = ptr_reg;
        end
    end
    
    assign parent_wdata = ptr_reg;
    assign done         = (state == ST_DONE);

endmodule