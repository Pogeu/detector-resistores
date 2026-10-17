function [labels, count] = bwlabel(BW, connectivity)
if nargin < 2
    connectivity = 8;
end
binary = logical(BW);
[height, width] = size(binary);
labels = zeros(height, width);
count = 0;
if connectivity == 4
    offsets = [-1 0; 1 0; 0 -1; 0 1];
else
    offsets = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
end

for y = 1:height
    for x = 1:width
        if ~binary(y, x) || labels(y, x) ~= 0
            continue;
        end
        count = count + 1;
        queue = zeros(height * width, 2);
        head = 1;
        tail = 1;
        queue(1, :) = [y x];
        labels(y, x) = count;
        while head <= tail
            current_y = queue(head, 1);
            current_x = queue(head, 2);
            head = head + 1;
            for k = 1:size(offsets, 1)
                next_y = current_y + offsets(k, 1);
                next_x = current_x + offsets(k, 2);
                if next_y >= 1 && next_y <= height && next_x >= 1 && next_x <= width && binary(next_y, next_x) && labels(next_y, next_x) == 0
                    labels(next_y, next_x) = count;
                    tail = tail + 1;
                    queue(tail, :) = [next_y next_x];
                end
            end
        end
    end
end
end
