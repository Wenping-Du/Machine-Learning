% # train the svm in the primal
function svm_model = svm_train_primal(data_train, label_train, regularisation_para_C)
   
    [number, dimension] = size(data_train);
    
%   cvx 
    cvx_begin 
    cvx_solver sedumi;
        variables w(dimension) b e(number)
        % w sqrt
        minimize(0.5 * w' * w + regularisation_para_C / number * sum(e)) 
        subject to
            label_train .* (data_train * w + b) >= 1 - e;
            e >= 0; 
    cvx_end
    
    svm_model.w = w;
    svm_model.b = b; 
    fprintf("b is %f\n", b); 
    fprintf("w is %f\n", mean(w)); 
    
end