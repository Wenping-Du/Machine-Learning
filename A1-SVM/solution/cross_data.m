% get crossdata
function [tra_set_x, tra_set_y, val_set_x, val_set_y] = cross_data(train_x, train_y, i, K)
    [number, dimension] = size(train_x);
    start_number = number / K * (i - 1) + 1;
    end_number = number / K * i;
    
    val_set_x = train_x(start_number : end_number, :);
    val_set_y = train_y(start_number : end_number, :);
    
    tra_set_x = train_x([1: start_number, end_number: end], :);
    tra_set_y = train_y([1: start_number, end_number: end], :);