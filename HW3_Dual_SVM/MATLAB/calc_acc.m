function acc = calc_acc(y_hat, y)
    correct = y_hat - y;
    num_corr = length(find(~correct));
    acc = num_corr/length(y);
end