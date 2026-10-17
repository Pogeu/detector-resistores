function output = imcomplement(input_image)
if islogical(input_image)
    output = ~input_image;
else
    output = max(input_image(:)) - input_image;
end
end
