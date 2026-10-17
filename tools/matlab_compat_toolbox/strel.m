function element = strel(shape, radius, varargin)
% Minimal structuring-element object for the package's binary operations.
if strcmpi(shape, 'square')
    element.Neighborhood = true(radius, radius);
elseif strcmpi(shape, 'disk')
    axis = -radius:radius;
    [xx, yy] = meshgrid(axis, axis);
    element.Neighborhood = (xx.^2 + yy.^2) <= radius^2;
else
    error('Forma strel nao suportada: %s', shape);
end
end
