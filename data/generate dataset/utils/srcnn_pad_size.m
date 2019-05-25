function [pad_h, pad_w] = srcnn_pad_size(input_size)
    
    [h_o, w_o] = srcnn_output_size(input_size);
    
    pad_h = (input_size(1) - h_o) / 2;
    pad_w = (input_size(2) - w_o) / 2;
    
end