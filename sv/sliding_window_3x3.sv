module sliding_window_3x3 #(
    parameter int IMG_WIDTH = 1920
)(
    input  logic              clk,
    input  logic              rst_n,
    
    input  logic              s_valid,
    input  logic              s_data,
    input  logic              s_user,
    input  logic              s_last,
    
    output logic              m_valid,
    output logic [2:0][2:0]   window,
    output logic              m_user,
    output logic              m_last
);

    logic line_buf_0 [0:IMG_WIDTH-1];
    logic line_buf_1 [0:IMG_WIDTH-1];
    
    logic user_buf_0 [0:IMG_WIDTH-1];
    logic last_buf_0 [0:IMG_WIDTH-1];
    
    logic [11:0] wr_ptr;
    
    logic rdata_0, rdata_1;
    logic ruser_0, rlast_0;
    
    assign rdata_0 = line_buf_0[wr_ptr];
    assign rdata_1 = line_buf_1[wr_ptr];
    assign ruser_0 = user_buf_0[wr_ptr];
    assign rlast_0 = last_buf_0[wr_ptr];

    // ==========================================
    // Block 1: Line Buffer Write Logic (BRAM Inference)
    // ==========================================
    always_ff @(posedge clk) begin
        if (s_valid) begin
            line_buf_0[wr_ptr] <= s_data;
            line_buf_1[wr_ptr] <= rdata_0;
            
            user_buf_0[wr_ptr] <= s_user;
            last_buf_0[wr_ptr] <= s_last;
        end
    end

    // ==========================================
    // Block 2: Write Pointer Counter
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (s_valid) begin
            if (wr_ptr == IMG_WIDTH - 1)
                wr_ptr <= '0;
            else
                wr_ptr <= wr_ptr + 1;
        end
    end

    logic [1:0] valid_sr;
    logic [1:0] user_sr;
    logic [1:0] last_sr;

    // ==========================================
    // Block 3: Shift Registers & Window Formation
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window   <= '0;
            valid_sr <= '0;
            user_sr  <= '0;
            last_sr  <= '0;
        end else if (s_valid) begin
            window[0][0] <= s_data;
            window[0][1] <= window[0][0];
            window[0][2] <= window[0][1];
            
            window[1][0] <= rdata_0;
            window[1][1] <= window[1][0];
            window[1][2] <= window[1][1];
            
            window[2][0] <= rdata_1;
            window[2][1] <= window[2][0];
            window[2][2] <= window[2][1];
            
            valid_sr <= {valid_sr[0], 1'b1};
            user_sr  <= {user_sr[0], ruser_0};
            last_sr  <= {last_sr[0], rlast_0};
        end else begin
            valid_sr <= {valid_sr[0], 1'b0};
        end
    end

    assign m_valid = valid_sr[1];
    assign m_user  = user_sr[1];
    assign m_last  = last_sr[1];

endmodule