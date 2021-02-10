% # data_train
function [data_x, data_y] = get_data(data_path)
%   read dataset
    data = csvread(data_path);
    [num, dim] = size(data);
    
%   get X 
    data_x = data(:, 2:dim);
    
%   get Y
    data_y = data(:, 1);
    data_y(data_y == 0) = -1;
end