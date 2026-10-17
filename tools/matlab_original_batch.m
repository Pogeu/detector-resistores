function matlab_original_batch(input_dir, output_json)
% Headless runner for the original MATLAB File Exchange implementation.
% It keeps the original segmentImageN, ResCheck, ResCheck2 and bandfind code.

package_dir = fileparts(input_dir);
tools_dir = fileparts(mfilename('fullpath'));
compat_dir = fullfile(tools_dir, 'matlab_compat_toolbox');
if exist('imbinarize', 'file') ~= 2 || exist('strel', 'file') ~= 2 || exist('regionprops', 'file') ~= 2
    addpath(compat_dir, '-begin');
    execution_mode = 'compatibility-shims-no-image-processing-toolbox';
else
    execution_mode = 'native-image-processing-toolbox';
end
addpath(package_dir);

files = dir(fullfile(input_dir, '*.*'));
files = files(~[files.isdir]);
valid = false(size(files));
for k = 1:numel(files)
    [~, ~, ext] = fileparts(files(k).name);
    valid(k) = any(strcmpi(ext, {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}));
end
files = files(valid);

records = struct('image', {}, 'candidate_count', {}, 'accepted_count', {}, ...
    'values', {}, 'bands_digits', {}, 'error', {}, 'engine', {});

for k = 1:numel(files)
    record.image = files(k).name;
    record.candidate_count = 0;
    record.accepted_count = 0;
    record.values = [];
    record.bands_digits = [];
    record.error = '';
    record.engine = execution_mode;
    try
        Imain = imread(fullfile(input_dir, files(k).name));
        Igray = rgb2gray(Imain);
        [BWmain, ~] = segmentImageN(Igray);
        hsv_image = rgb2hsv(Imain);
        avgV = mean(hsv_image(: ,: ,3), 'all');

        if avgV < .9
            se = strel('square', 41);
        else
            se = strel('square', 18);
        end
        Iopenned = imopen(BWmain, se);
        labeled = bwlabel(Iopenned, 4);
        stats = regionprops(labeled, 'Eccentricity', 'Area', 'BoundingBox');
        eccentricities = [stats.Eccentricity];
        idxOfResistors = find(eccentricities);
        record.candidate_count = numel(idxOfResistors);

        resimg = {};
        for i = 1:numel(idxOfResistors)
            bbox = stats(idxOfResistors(i)).BoundingBox;
            candidate = imcrop(Imain, bbox);
            [BWres, ~] = ResCheck(candidate);
            Iopennedres = imopen(BWres, strel('disk', 6));
            [~, numObjectsRes] = bwlabel(Iopennedres, 4);
            accepted = numObjectsRes ~= 0;
            if ~accepted
                [BWres, ~] = ResCheck2(candidate);
                Iopennedres = imopen(BWres, strel('disk', 2));
                [~, numObjectsRes] = bwlabel(Iopennedres, 4);
                accepted = numObjectsRes ~= 0;
            end
            if accepted
                resimg{end + 1} = candidate; %#ok<AGROW>
            end
        end

        record.accepted_count = numel(resimg);
        values = zeros(1, numel(resimg));
        bands_digits = zeros(numel(resimg), 3);
        for r = 1:numel(resimg)
            run(fullfile(package_dir, 'bandfind.m'));
            values(r) = (Band1 * 10 + Band2) * 10^Band3;
            bands_digits(r, :) = [Band1 Band2 Band3];
        end
        record.values = values;
        record.bands_digits = bands_digits;
    catch exception
        record.error = exception.message;
    end
    records(end + 1) = record; %#ok<AGROW>
end

fid = fopen(output_json, 'w');
if fid < 0
    error('Nao foi possivel abrir o arquivo de saida: %s', output_json);
end
fwrite(fid, jsonencode(records), 'char');
fclose(fid);
fprintf('MATLAB original (%s): %d imagens processadas -> %s\n', execution_mode, numel(records), output_json);
end
