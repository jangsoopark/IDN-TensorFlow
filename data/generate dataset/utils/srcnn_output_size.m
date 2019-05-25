
function [h, w] = srcnn_output_size(input_size)
    h = (input_size(1) - 9) + 1;
    h = (h - 1) + 1;
    h = (h - 5) + 1;
    
    w = (input_size(2) - 9) + 1;
    w = (w - 1) + 1;
    w = (w - 5) + 1;
end