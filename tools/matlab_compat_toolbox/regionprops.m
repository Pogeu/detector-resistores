function stats = regionprops(input_image, varargin)
properties = varargin;
if islogical(input_image) || all(input_image(:) == 0 | input_image(:) == 1)
    [labels, count] = bwlabel(input_image, 8);
else
    labels = input_image;
    count = max(labels(:));
end
stats = repmat(struct(), 1, count);
for label = 1:count
    [rows, cols] = find(labels == label);
    if isempty(rows)
        continue;
    end
    values = struct();
    values.Area = numel(rows);
    values.Centroid = [mean(cols) mean(rows)];
    min_col = min(cols);
    max_col = max(cols);
    min_row = min(rows);
    max_row = max(rows);
    values.BoundingBox = [min_col - 0.5 min_row - 0.5 max_col - min_col + 1 max_row - min_row + 1];
    if numel(rows) < 2
        values.Eccentricity = 0;
    else
        centered = [double(cols) - mean(cols), double(rows) - mean(rows)];
        covariance = (centered' * centered) / size(centered, 1);
        eigenvalues = eig(covariance);
        major = max(eigenvalues);
        minor = min(eigenvalues);
        if major <= eps
            values.Eccentricity = 0;
        else
            values.Eccentricity = sqrt(max(0, 1 - minor / major));
        end
    end
    for property_index = 1:numel(properties)
        property_name = properties{property_index};
        property_key = lower(property_name);
        if strcmp(property_key, 'centroid')
            property_value = values.Centroid;
        elseif strcmp(property_key, 'boundingbox')
            property_value = values.BoundingBox;
        elseif strcmp(property_key, 'eccentricity')
            property_value = values.Eccentricity;
        elseif strcmp(property_key, 'area')
            property_value = values.Area;
        else
            error('Propriedade regionprops nao suportada: %s', property_name);
        end
        stats(label).(property_name) = property_value;
    end
end
end
