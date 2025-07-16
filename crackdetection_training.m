clc; clear;

%% 🔹 Folder Setup
baseFolder = 'C:\Users\tallu\Downloads\dataverse_files\dataset 1';
posFolder  = fullfile(baseFolder, 'Positive');
negFolder  = fullfile(baseFolder, 'negative');
imgFolder  = fullfile(baseFolder, 'processed', 'images');
maskFolder = fullfile(baseFolder, 'processed', 'masks');

if ~exist(imgFolder, 'dir'); mkdir(imgFolder); end
if ~exist(maskFolder, 'dir'); mkdir(maskFolder); end

%% 🔹 Initialize
imgIdx = 1;

% Process POSITIVE images: generate rough masks
posImages = dir(fullfile(posFolder, '*.jpg'));
for i = 1:length(posImages)
    img = imread(fullfile(posFolder, posImages(i).name));
    gray = rgb2gray(img);
    bw = imbinarize(imadjust(gray));             % basic threshold
    bw = bwareaopen(bw, 30);                     % remove small noise
    bw = imdilate(bw, strel('disk', 1));         % enhance crack edges
    
    % Save image and mask
    imgName = sprintf('img_%04d.jpg', imgIdx);
    maskName = sprintf('img_%04d.png', imgIdx);
    imwrite(img, fullfile(imgFolder, imgName));
    imwrite(uint8(bw) * 255, fullfile(maskFolder, maskName));
    imgIdx = imgIdx + 1;
end

% Process NEGATIVE images: blank masks
negImages = dir(fullfile(negFolder, '*.jpg'));
for i = 1:length(negImages)
    img = imread(fullfile(negFolder, negImages(i).name));
    [H, W, ~] = size(img);
    blankMask = uint8(zeros(H, W));
    
    imgName = sprintf('img_%04d.jpg', imgIdx);
    maskName = sprintf('img_%04d.png', imgIdx);
    imwrite(img, fullfile(imgFolder, imgName));
    imwrite(blankMask, fullfile(maskFolder, maskName));
    imgIdx = imgIdx + 1;
end

%% 🔹 Prepare Dataset
allImgs = dir(fullfile(imgFolder, '*.jpg'));
imgPaths = fullfile(imgFolder, {allImgs.name});
maskPaths = fullfile(maskFolder, replace({allImgs.name}, '.jpg', '.png'));

labels = cellfun(@(p) max(imread(p), [], 'all') > 0, maskPaths); % 1 = positive
posIdx = find(labels);
negIdx = find(~labels);

rng(42);
valCount = round(0.2 * numel(posIdx));
valIdx = posIdx(randperm(numel(posIdx), valCount));
trainIdx = setdiff(1:numel(imgPaths), valIdx);

%% 🔹 Datastore Setup
classes = ["background", "crack"];
labelIDs = [0, 255];

imdsTrain = imageDatastore(imgPaths(trainIdx));
pxdsTrain = pixelLabelDatastore(maskPaths(trainIdx), classes, labelIDs);
imdsVal   = imageDatastore(imgPaths(valIdx));
pxdsVal   = pixelLabelDatastore(maskPaths(valIdx), classes, labelIDs);

%% 🔹 Augment + Resize
imageSize = [224 224 3];
augmenter = imageDataAugmenter('RandXReflection', true, 'RandRotation', [-10 10]);
trainingData = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'OutputSize', imageSize(1:2), 'DataAugmentation', augmenter);
validationData = pixelLabelImageDatastore(imdsVal, pxdsVal, ...
    'OutputSize', imageSize(1:2));

%% 🔹 Define & Train Model
try
    lgraph = deeplabv3plusLayers(imageSize, numel(classes), 'resnet18');
catch
    error('⚠️ Install Deep Learning Toolbox Model for ResNet-18.');
end

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',64, ...
    'MiniBatchSize',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationData, ...
    'ValidationFrequency',10, ...
    'VerboseFrequency',5, ...
    'Plots','training-progress');

fprintf('\n🚀 Training started...\n');
net = trainNetwork(trainingData, lgraph, options);

%% 🔹 Evaluate Model
fprintf('\n📊 Evaluating...\n');
pxdsPred = semanticseg(imdsVal, net, 'MiniBatchSize',4, 'OutputType','uint8');
metrics = evaluateSemanticSegmentation(pxdsPred, pxdsVal, 'Verbose', false);
disp(metrics.DataSetMetrics);
disp(metrics.ClassMetrics);

confMat = table2array(metrics.ConfusionMatrix);
figure;
confusionchart(confMat, classes, 'Title', 'Confusion Matrix', ...
    'RowSummary','row-normalized', 'ColumnSummary','column-normalized');

%% 🔹 Save Model
modelPath = fullfile(baseFolder, 'trainedCrackModel.mat');
save(modelPath, 'net');
fprintf('\n✅ Model saved at: %s\n', modelPath);
