clc; clear;

%% 🔹 Step 1: Setup Paths
projectFolder = 'C:\Users\tallu\Downloads\dataverse_files\dataset 1';  % ✅ Update as needed
imgFolder     = fullfile(projectFolder, 'images');
jsonFolder    = fullfile(projectFolder, 'json');
maskFolder    = fullfile(projectFolder, 'masks');

if ~exist(maskFolder, 'dir')
    mkdir(maskFolder);
end

%% 🔹 Step 2: Convert JSON to Masks
jsonFiles = dir(fullfile(jsonFolder, '*.json'));

for k = 1:length(jsonFiles)
    jsonPath = fullfile(jsonFolder, jsonFiles(k).name);
    imageName = replace(jsonFiles(k).name, '.json', '.jpg');
    imagePath = fullfile(imgFolder, imageName);

    if ~isfile(imagePath)
        fprintf('❌ Missing image: %s\n', imageName);
        continue;
    end

    img = imread(imagePath);
    [H, W, ~] = size(img);

    txt = fileread(jsonPath);
    try
        jsonData = jsondecode(txt);
    catch
        fprintf('⚠️ Failed to decode JSON: %s\n', jsonFiles(k).name);
        continue;
    end

    mask = false(H, W);

    if isfield(jsonData, 'shapes')
        for s = 1:length(jsonData.shapes)
            shape = jsonData.shapes(s);
            if isfield(shape, 'points')
                points = shape.points;
                x = cellfun(@(p) p(1), num2cell(points, 2));
                y = cellfun(@(p) p(2), num2cell(points, 2));
                polyMask = poly2mask(x, y, H, W);
                mask = mask | polyMask;
            end
        end
    end

    maskName = replace(jsonFiles(k).name, '.json', '.png');
    imwrite(uint8(mask) * 255, fullfile(maskFolder, maskName));
    fprintf('✅ Saved mask: %s\n', maskName);
end

%% 🔹 Step 3: Prepare Datastores
classes = ["background", "crack"];
labelIDs = [0, 255];

imds = imageDatastore(imgFolder);
pxds = pixelLabelDatastore(maskFolder, classes, labelIDs);

% 🔸 Split 80% training, 20% validation
numFiles = numel(imds.Files);
rng(1);
shuffledIdx = randperm(numFiles);
numTrain = round(0.8 * numFiles);

trainIdx = shuffledIdx(1:numTrain);
valIdx   = shuffledIdx(numTrain+1:end);

imdsTrain = imageDatastore(imds.Files(trainIdx));
pxdsTrain = pixelLabelDatastore(pxds.Files(trainIdx), classes, labelIDs);

imdsVal   = imageDatastore(imds.Files(valIdx));
pxdsVal   = pixelLabelDatastore(pxds.Files(valIdx), classes, labelIDs);

% 🔸 Create pixelLabelImageDatastore
imageSize = [224 224 3];
augmenter = imageDataAugmenter();
trainingData = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'OutputSize', imageSize(1:2), ...
    'DataAugmentation', augmenter);

validationData = pixelLabelImageDatastore(imdsVal, pxdsVal, ...
    'OutputSize', imageSize(1:2));

%% 🔹 Step 4: Define DeepLabV3+ Network
try
    lgraph = deeplabv3plusLayers(imageSize, numel(classes), 'resnet18');
catch
    error(['🚫 Could not load ResNet-18. Install: ' ...
           'Deep Learning Toolbox Model for ResNet-18.']);
end

%% 🔹 Step 5: Set Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 10, ...
    'VerboseFrequency', 5, ...
    'Plots', 'training-progress');

%% 🔹 Step 6: Train the Model
fprintf('\n🚀 Starting training...\n');
net = trainNetwork(trainingData, lgraph, options);

%% 🔹 Step 7: Evaluate on Validation Set
fprintf('\n📊 Evaluating on validation set...\n');
pxdsResults = semanticseg(imdsVal, net, 'MiniBatchSize', 4, 'OutputType', 'uint8');

metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal, 'Verbose', false);
disp(metrics.DataSetMetrics);
disp(metrics.ClassMetrics);

% 🔸 Confusion Matrix
confMat = table2array(metrics.ConfusionMatrix);  % Convert from table
figure;
confusionchart(confMat, classes, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% 🔹 Step 8: Save Trained Network
modelPath = fullfile(projectFolder, 'trainedCrackSegModel.mat');
save(modelPath, 'net');
fprintf('✅ Model saved to: %s\n', modelPath);
