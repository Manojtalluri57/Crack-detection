clc; clear;

%% 🔹 Load Trained Model
imageFolder = 'C:\Users\tallu\Downloads\dataverse_files\testing 1';  % Test images folder
outputFolder = fullfile(imageFolder, 'output');                      % Output folder
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end

modelPath = 'C:\Users\tallu\Downloads\dataverse_files\dataset 1\trainedCrackModel.mat';
load(modelPath, 'net');
fprintf("📦 Model loaded from: %s\n", modelPath);

%% 🔹 Load Test Images
imgFiles = dir(fullfile(imageFolder, '*.jpg'));  % Adjust if PNG
if isempty(imgFiles)
    error('❌ No test images found in folder: %s', imageFolder);
end

imgPaths = fullfile({imgFiles.folder}, {imgFiles.name});
inputSize = net.Layers(1).InputSize(1:2);

%% 🔹 Predict and Measure (in pixels)
allResults = struct();
for i = 1:numel(imgPaths)
    img = imread(imgPaths{i});
    if size(img,3) ~= 3
        img = repmat(img, [1 1 3]);
    end
    imgResized = imresize(img, inputSize);

    % Predict segmentation
    pred = semanticseg(imgResized, net);
    crackMask = pred == "crack";

    % Enhancement + cleanup (4-step processing)
    gray = rgb2gray(imgResized);
    enhanced = adapthisteq(gray);
    thresh = imbinarize(enhanced, 'adaptive');
    clean = bwareaopen(thresh, 30);
    finalMask = crackMask & clean;

    % Width estimation (px)
    skeleton = bwmorph(finalMask, 'skel', Inf);
    dist = bwdist(~finalMask);
    width_px = 2 * dist(skeleton);
    maxWidth_px = max(width_px(:));
    if isempty(maxWidth_px); maxWidth_px = 0; end

    allResults(i).imgName = imgFiles(i).name;
    allResults(i).maxWidth_px = maxWidth_px;
    allResults(i).original = imgResized;
    allResults(i).mask = finalMask;
end

fprintf("\n📂 Predictions done. Starting calibration...\n");

%% 🔹 Calibration (Interactive for First 3)
knownWidths_mm = [];
measuredWidths_px = [];
count = 0;
i = 1;

while count < 3 && i <= numel(allResults)
    if allResults(i).maxWidth_px > 0
        fprintf('Image %d: %s\n', i, allResults(i).imgName);
        real_mm = input('👉 Enter real crack width in mm: ');
        knownWidths_mm(end+1) = real_mm;
        measuredWidths_px(end+1) = allResults(i).maxWidth_px;
        count = count + 1;
    end
    i = i + 1;
end

if count < 3
    error('❌ Not enough valid crack images for calibration.');
end

pxPerMM_values = measuredWidths_px ./ knownWidths_mm;
pxPerMM = mean(pxPerMM_values);
fprintf('\n🔧 Calibrated pxPerMM = %.2f\n', pxPerMM);

%% 🔹 Final Output with Comparison Images
for i = 1:numel(allResults)
    width_px = allResults(i).maxWidth_px;
    width_mm = width_px / pxPerMM;

    % Severity classification
    if width_mm < 0.1
        severity = "Hairline";
    elseif width_mm < 0.3
        severity = "Minor";
    elseif width_mm < 0.7
        severity = "Moderate";
    elseif width_mm < 2.0
        severity = "Severe";
    else
        severity = "Critical";
    end

    % Overlay mask
    overlay = labeloverlay(allResults(i).original, uint8(allResults(i).mask), 'Transparency', 0.4);

    % Add text
    textStr = sprintf('Width: %.2f mm | %s', width_mm, severity);
    annotatedOverlay = insertText(overlay, [10, 10], textStr, ...
        'FontSize', 18, 'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.6);

    % Create side-by-side comparison
    comparison = [allResults(i).original, annotatedOverlay];

    % Save output
    baseName = allResults(i).imgName(1:end-4);
    outPath = fullfile(outputFolder, [baseName '_comparison.png']);
    imwrite(comparison, outPath);

    % Console info
    fprintf("✅ %s → %.2f mm (%s)\n", allResults(i).imgName, width_mm, severity);
end

fprintf("\n📁 Comparison images saved to: %s\n", outputFolder);
