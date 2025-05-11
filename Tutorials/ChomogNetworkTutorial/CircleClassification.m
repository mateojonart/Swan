clc;
clear;
close all;

%% Initialization of hyperparameters
pol_deg         = 1;
testratio       = 20;
lambda          = 0.0;
learningRate    = 0.0001;
hiddenLayers    = 128 * ones(1, 4);

%% INITIALIZATION
% Store dataset file name
s.fileName = 'DataCirclesShort.csv';

% Load model parameters
s.polynomialOrder = pol_deg;
s.testRatio       = testratio;
s.networkParams.hiddenLayers    = hiddenLayers;
s.optimizerParams.learningRate  = learningRate;
s.optimizerParams.maxEpochs = 500; % 1000 is the best option, but we use 10 to pass the tutorial quickly
s.costParams.lambda             = lambda;
s.costParams.costType           = '-loglikelihood';

s.networkParams.HUtype = 'ReLU'; % capas internas
s.networkParams.OUtype = 'softmax'; % ultima capa

% Select the model's features
s.xFeatures = [1, 2];
s.yFeatures = [3, 4, 5, 6];
% cHomogIdxs = [11, 12, 22, 33];

% Load data
%data   = cHomogData(s);
data   = Data(s);
s.data = data;

% Train the model
opt = OptimizationProblemNN(s);
opt.solve();
opt.plotCostFnc();

%%

X = [0.2, 0.6];
Xful = createFeatures(X, pol_deg);
Y = opt.computeOutputValues(Xful);
%dY = opt.computeGradient([0.95, 0.05]);

disp(Y)



function Xful = createFeatures(X, maxOrder)
    x1 = X(:, 1);
    x2 = X(:, 2);
    capacity = (maxOrder + 1) * (maxOrder + 2) / 2 - 1;
    Xful = [zeros(size(x1, 1), capacity)];
    cont = 1;
    for ii = 1:maxOrder
        for jj = 0:ii
            Xful(:,cont) = x2.^(jj).*x1.^(ii-jj);
            cont = cont+1;
        end
    end
end