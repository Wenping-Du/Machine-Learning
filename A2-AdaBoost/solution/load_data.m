% # data_train
function [X, Y, train_x, train_y, test_x, test_y] = load_data(data_path)
%   read dataset
    dataset = importdata(data_path);
    X = dataset.data;
    [~, ~, Y] = unique(dataset.textdata(:,2));
    Y(Y == 2) = -1;
    
%   get X 
    train_x = X(1:300, :);
    test_x = X(301:end, :);
    
%   get Y
    train_y  = Y(1:300, :);
    test_y = Y(301:end, :);
end