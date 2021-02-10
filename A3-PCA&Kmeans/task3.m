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
plot(xPos, yPos,'b'); % train acc
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
    % task3: Initial random centroids
    centroids = zeros(K, 784);
    for i = 1 : K
        centroids(i, :) = train_x(randi(total_x), :);
    end
    
    pred_distance = 0;
    for p = 1 : iteration
        distance = zeros(K, 1);
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
            distance(class) = distance(class) + min_d;
            min_ds(i) = min_d;
        end
        % Reset center
        acc = 0;
        for k = 1 : K
            centroids(k, :)= mean(train_x(pre_class == k-1,:)); % New cluster centers in x
            acc = acc + sum(train_y(pre_class == k-1,:) == k-1)/600;
        end
%         if p >= 3
%             k_loss(p) = abs(sum(sum(d)) - pred_distance) / 10;
%         end
        
        k_acc(p) = acc / 10;
%         if sum(sum(d)) == pred_distance
%             break;
%         end
%         pred_distance = sum(sum(d));
        k_loss(p) = mean(min_ds)/dim;
    end

end