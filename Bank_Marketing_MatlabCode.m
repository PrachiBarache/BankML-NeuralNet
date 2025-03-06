clc;
% Set VariableNamingRule to 'preserve'
opts = detectImportOptions('X_train_processed.csv');
opts.VariableNamingRule = 'preserve';

% Load the preprocessed datasets
X_train = readtable('X_train_processed.csv');
y_train = readtable('y_train_processed.csv');
X_test = readtable('X_test_processed.csv');
y_test = readtable('y_test_processed.csv');

% Extract features and targets from the datasets
x_train = X_train{:,:}';
t_train = y_train{:,:}';
x_test = X_test{:,:}';
t_test = y_test{:,:}';

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = 128;
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
tic;
[net,tr] = train(net,x_train,t_train);
training_time = toc;

% Test the Network
y_train = net(x_train);
train_accuracy = 1 - perform(net,t_train,y_train);
y_test = net(x_test);
test_accuracy = 1 - perform(net,t_test,y_test);

% View the Network
view(net)



% Display Results Summary
fprintf('Results Summary:\n');
fprintf('   Hidden Size  Learning Rate  Training Time (s)  Train Accuracy  Test Accuracy\n');
fprintf('------------------------------------------------------------------------------\n');
fprintf('           %d           %s             %.6f           %.6f         %.6f\n', hiddenLayerSize, trainFcn, training_time, train_accuracy, test_accuracy);


