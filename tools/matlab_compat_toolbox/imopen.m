function output = imopen(BW, element)
kernel = logical(element.Neighborhood);
output = matlab_compat_morph(BW, kernel, 'open');
end
