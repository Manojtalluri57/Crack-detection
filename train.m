clc; clear;

%% 🔹 Paths Setup
projectFolder = 'C:\Users\tallu\Downloads\dataverse_files\dataset 1';
posImgFolder  = fullfile(projectFolder, 'images', 'positive');
negImgFolder  = fullfile(projectFolder, 'images', 'negative');
posMaskFolder = fullfile(projectFolder, 'masks', 'positive');
negMaskFolder = fullfile(projectFolder, 'masks', 'negative');
allMaskFolder = fullfile(projectFolder, 'masks', 'all');

if ~exist(allMaskFolder, 'dir')
    mkdir(allMaskFolder);
end

%% 🔹 Generate Blank Masks for Negative Images
negImages = dir(fullfile(negImgFolder, '*.jpg'));
for k = 1:length(negImages)
    img = imread(fullfile(negImgFolder, negImages(k).name));
    [H, W, ~] = size(img);
    blankMask = uint8(zeros(H, W));  % All background (label 0)
    maskName = replace(negImages(k).name, '.jpg', '.png');
    imwrite(blankMask, fullfile(allMaskFolder, maskName));
end

%% 🔹 Copy Positive Masks to allMaskFolder
posImages = dir(fullfile(posImgFolder, '*.jpg'));
for k = 1:length(posImages)
    maskName = replace(posImages(k).name, '.jpg', '.png');
    srcMaskPath = fullfile(posMaskFolder, maskName);
    dstMaskPath = fullfile(allMaskFolder, maskName);
    if exist(srcMaskPath, 'file')
        copyfile(srcMaskPath, dstMaskPath);
    else
        warning('❌ Missing mask for: %s', posImages(k).name);
    end
end

%% 🔹 Combine All Image and Mask Paths
allPos = dir(fullfile(posImgFolder, '*.jpg'));
allNeg = dir(fullfile(negImgFolder, '*.jpg'));
allImages = [allPos; allNeg];

imgPaths  = fullfile({allImages.folder}, {allImages.name})';
maskNames = replace({allImages.name}, '.jpg', '.png');
maskPaths = fullfile(allMaskFolder, maskNames)';

% Check if all masks exist
missingMasks = ~isfile(maskPaths);
if any(missingMasks)
    error('⚠️ Missing masks for %d image(s). Check your mask folder.', sum(missingMasks));
end

%% 🔹 Define Classes and Datastores
classes = ["background", "crack"];
labelIDs = [0, 255];

imds = imageDatastore(imgPaths);
pxds = pixelLabelDatastore(maskPaths, classes, labelIDs);

%% 🔹 Shuffle and Split
numFiles = numel(imds.Files);
rng(42);  % Seed for reproducibility
indices = randperm(numFiles);
nTrain = round(0.8 * numFiles);
trainIdx = indices(1:nTrain);
valIdx   = indices(nTrain+1:end);

imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds, trainIdx);
imdsVal   = subset(imds, valIdx);
pxdsVal   = subset(pxds, valIdx);

%% 🔹 Augmentation and Datastores
imageSize = [224 224 3];
augmenter = imageDataAugmenter('RandXReflection', true, 'RandRotation', [-10 10]);

trainingData = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'OutputSize', imageSize(1:2), 'DataAugmentation', augmenter);
validationData = pixelLabelImageDatastore(imdsVal, pxdsVal, ...
    'OutputSize', imageSize(1:2));

%% 🔹 DeepLabV3+ Model
try
    lgraph = deeplabv3plusLayers(imageSize, numel(classes), 'resnet18');
catch
    error('❌ DeepLabV3+ with ResNet-18 requires Deep Learning Toolbox Model for ResNet-18.');
end

%% 🔹 Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 10, ...
    'VerboseFrequency', 5, ...
    'Plots', 'training-progress');

fprintf('\n🚀 Training started...\n');
net = trainNetwork(trainingData, lgraph, options);

%% 🔹 Evaluation
fprintf('\n📊 Evaluating model...\n');
pxdsPred = semanticseg(imdsVal, net, 'MiniBatchSize', 4, 'OutputType', 'uint8');
metrics = evaluateSemanticSegmentation(pxdsPred, pxdsVal, 'Verbose', false);
disp(metrics.DataSetMetrics);
disp(metrics.ClassMetrics);

% Confusion Matrix
confMat = table2array(metrics.ConfusionMatrix);
figure;
confusionchart(confMat, classes, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

%% 🔹 Save Trained Model
modelPath = fullfile(projectFolder, 'trainedCrackSegModel.mat');
save(modelPath, 'net');
fprintf('✅ Model saved to: %s\n', modelPath);

