% Kmeans
[train_y, train_x] = get_data('mnist_train.csv');
[test_y, test_x] = get_data('mnist_test.csv');
train_y(train_y == -1) = 0;

iteration = 100;
[loss, acc] = kmeans(iteration, train_x, train_y);

figure
hold on
xPos = acc;
yPos = 'DisplayName';
plot(xPos ,yPos,'b'); % train acc
grid on
title('Kmeans ACC Curves','Color','b');
xlabel('Iterations');
ylabel('Accuracy');
legend('Training acc');
box on
hold off

figure
hold on
xPos = loss;
yPos = 'DisplayName';
plot(xPos, yPos,'b'); % train loss
grid on
title('Kmeans Loss Curves','Color','b');
xlabel('Iterations');
ylabel('Loss');
legend('Training loss');
box on
hold off


function [k_loss,k_acc] = kmeans(iteration, train_x, train_y)
    K = 10;
    total_x = 6000;
    [~, dim] = size(train_x);
    % task4: Initial random centroids u1,u2,...uk from different classes
    centroids = zeros(K,784);
    for i = 1 : K
        stack = train_x(train_y == i - 1, :);
        [s, ~] = size(stack);
        centroids(i,:) = stack(randi(s),:);
    end
    
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
            pre_class(i) = class - 1;
            % update distance matrix
            dist(class) = dist(class) + min_d;
            min_ds(i) = min_d;
        end
        % Reset center
        acc = 0;
        for k = 1 : K
            centroids(k, :)= mean(train_x(pre_class == k-1,:)); % New cluster centers in x
            acc = acc + sum(train_y(pre_class == k-1,:) == k-1)/600;
        end
        k_acc(p) = acc / 10;
        k_loss(p) = mean(min_ds)/dim;
    end

end