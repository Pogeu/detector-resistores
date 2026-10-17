function output = matlab_compat_morph(BW, kernel, operation)
binary = logical(BW);
if strcmp(operation, 'open')
    output = matlab_compat_dilate(matlab_compat_erode(binary, kernel), kernel);
elseif strcmp(operation, 'close')
    output = matlab_compat_erode(matlab_compat_dilate(binary, kernel), kernel);
else
    error('Operacao morfologica desconhecida: %s', operation);
end
end

function output = matlab_compat_erode(binary, kernel)
needed = sum(kernel(:));
output = conv2(double(binary), double(kernel), 'same') >= needed;
end

function output = matlab_compat_dilate(binary, kernel)
output = conv2(double(binary), double(kernel), 'same') > 0;
end
