function [pos] = rect_pos(x, y)

min_x = min(x(1), x(2));
max_x = max(x(1), x(2));

min_y = min(y(1), y(2));
max_y = max(y(1), y(2));

pos = [min_x, min_y, max_x - min_x, max_y - min_y];
end

