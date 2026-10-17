function output = imfill(BW, mode)
if ~strcmpi(mode, 'holes')
    error('Somente imfill(BW, ''holes'') e suportado.');
end
binary = logical(BW);
background = ~binary;
[height, width] = size(binary);
visited = false(height, width);
queue = zeros(height * width, 2);
head = 1;
tail = 0;

for x = 1:width
    [visited, queue, tail] = enqueue_if_background(1, x, background, visited, queue, tail);
    [visited, queue, tail] = enqueue_if_background(height, x, background, visited, queue, tail);
end
for y = 1:height
    [visited, queue, tail] = enqueue_if_background(y, 1, background, visited, queue, tail);
    [visited, queue, tail] = enqueue_if_background(y, width, background, visited, queue, tail);
end

while head <= tail
    y = queue(head, 1);
    x = queue(head, 2);
    head = head + 1;
    neighbors = [y-1 x; y+1 x; y x-1; y x+1];
    for k = 1:4
        yy = neighbors(k, 1);
        xx = neighbors(k, 2);
        if yy >= 1 && yy <= height && xx >= 1 && xx <= width && background(yy, xx) && ~visited(yy, xx)
            visited(yy, xx) = true;
            tail = tail + 1;
            queue(tail, :) = [yy xx];
        end
    end
end

holes = background & ~visited;
output = binary | holes;
end

function [visited, queue, tail] = enqueue_if_background(y, x, background, visited, queue, tail)
if background(y, x) && ~visited(y, x)
    visited(y, x) = true;
    tail = tail + 1;
    queue(tail, :) = [y x];
end
end
