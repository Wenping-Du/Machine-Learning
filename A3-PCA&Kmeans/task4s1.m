% Kmeans
[train_y, train_x] = get_data('mnist_train.csv');
[test_y, test_x] = get_data('mnist_test.csv');
train_y(train_y == -1) = 0;

% 2-256
dimension = 8;
iteration = 100;

result_loss = zeros(dimension, 1);
result_acc = zeros(dimension, 1);
% PCA dimension reduce
dimension_array = zeros(dimension, 1);
for i = 1:dimension
    dimension_array(i) = 2^i;
    [train_vector, train_mean] = task1(train_x, 2^i);
    res_train_x = (train_x - train_mean) * train_vector;
    [loss, acc] = kmeans(iteration, res_train_x, train_y);
    result_loss(i) = mean(loss);
    result_acc(i) = mean(acc);
end

figure
hold on
plot(dimension_array ,result_acc,'b'); % train acc
grid on
title('Kmeans Accuracy curves','Color','b');
xlabel('Dimensions');
ylabel('Accuracy');
% legend('Training acc');
box on
hold off


figure
hold on
plot(dimension_array, result_loss, 'b'); % train acc
grid on
title('Kmeans Loss curves','Color','b');
xlabel('Dimensions');
ylabel('Loss');
% legend('Training loss');
box on
hold off

fprintf('%d,', dimension_array);
fprintf('\nacc: \n');
fprintf('%f,', result_acc);
fprintf('\nloss: \n');
fprintf('%f,', result_loss);

function [k_loss, k_acc] = kmeans(iteration, train_x, train_y)
    K = 10;
    total_x = 6000;
    [~, dim] = size(train_x);
    % task4: Initial random centroids u1,u2,...uk from different classes
    centroids = zeros(K, dim);
    for i = 1 : K
        stack = train_x(train_y == i - 1, :);
        [s, ~] = size(stack);
        centroids(i,:) = stack(randi(s),:);
    end
%     pred_distance = 0;
    k_loss = [];
    for p = 1 : iteration
        dist = zeros(K, 1);
        min_ds = zeros(total_x, 1);
        for i = 1 : total_x
            for j = 1 : K
                % calculate the distance between Xj and ui
                d(i,j)=sqrt(sum((train_x(i, :)-centroids(j,:)).^2));
            end
            % get the minimum distance and index of the centroids
            [min_d, class] = min(d(i, :));
            pre_label(i) = class - 1;
            % update distance matrix
            dist(class) = dist(class) + min_d;
            min_ds(i) = min_d;
        end
        % fixed center
        acc = 0;
        for k = 1 : K
%             centroids(k, :)= mean(train_x(pre_label == k - 1,:)); % New cluster centers in x
            acc = acc + sum(train_y(pre_label == k - 1,:) == k - 1) / total_x * K;
        end
        k_acc(p) = acc / K;
        k_loss(p) = mean(min_ds)/dim;
    end

end