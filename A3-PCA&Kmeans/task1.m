% PCA
function [mean_vector, mean_x] = task1(X, dimension)
    [row, ~] = size(X);
    mean_x = mean(X);
    % 中心化
    X = X - mean_x;
    % 协方差矩阵
    covx = X' * X / row;
    % 特征值分解
    [V, D] = eig(covx);
    [~, index] = sort(diag(D),'descend');
    mean_vector = V(:, index(1:dimension));
end