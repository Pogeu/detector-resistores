function output = imclose(BW, element)
kernel = logical(element.Neighborhood);
output = matlab_compat_morph(BW, kernel, 'close');
end
