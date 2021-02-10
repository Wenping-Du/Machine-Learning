function [result] = adaboost(X, y, epochs)

    % initialise the weight distribution
    num = size(X, 1);
    D = ones(num, 1) / num;

    result = struct;
    result.alpha = zeros(epochs,1);
    result.dimen = zeros(epochs,1);
    result.s = zeros(epochs,1);
    result.direct = zeros(epochs,1);
    hxSum = zeros(size(y));
    %300*30
    [n, ~] = size(X);
    mask = - ones(n,n);
    for i = 1 : n
        for j = 1:i
            mask(i,j) = 1;
        end
    end
    for t = 1 : epochs
        %choose the best separator
        stump = build_stump2(X, y, D, mask);
        result(t).alpha = 0.5 * log((1-stump.weightError) / stump.weightError);
        D = D .* exp(-result(t).alpha .* y .* stump.hx);
        D = D ./ sum(D);
        %reverse the result
        result(t).dimen = stump.dimension;
        result(t).s = stump.threshold;
        result(t).direct = stump.direction;
        result(t).weightError = stump.weightError;
        result(t).hx = stump.hx;
        hxSum = hxSum + result(t).hx * result(t).alpha;
        Hx = sign(hxSum);
        result(t).errorRate = sum(Hx ~= y) / length(X);
    end
end