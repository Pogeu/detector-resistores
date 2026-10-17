function output = imcrop(image, bbox)
x0 = max(1, floor(bbox(1) + 0.5) + 1);
y0 = max(1, floor(bbox(2) + 0.5) + 1);
x1 = min(size(image, 2), x0 + floor(bbox(3)) - 1);
y1 = min(size(image, 1), y0 + floor(bbox(4)) - 1);
if x1 < x0 || y1 < y0
    output = image([]);
else
    output = image(y0:y1, x0:x1, :);
end
end
