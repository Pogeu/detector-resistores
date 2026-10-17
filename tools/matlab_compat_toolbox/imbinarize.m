function BW = imbinarize(X)
% Global Otsu threshold compatible with the default imbinarize(X) call.
values = double(X(:));
max_value = max(values);
if max_value <= 1
    scaled = round(values * 255);
else
    scaled = round(values / max_value * 255);
end
counts = accumarray(scaled + 1, 1, [256 1]);
probability = counts / sum(counts);
levels = (0:255)';
omega = cumsum(probability);
mu = cumsum(probability .* levels);
mu_total = mu(end);
between = (mu_total * omega - mu).^2 ./ max(omega .* (1 - omega), eps);
[~, index] = max(between);
threshold = (index - 1) / 255;
if max_value <= 1
    BW = X > threshold;
else
    BW = X > threshold * max_value;
end
end
