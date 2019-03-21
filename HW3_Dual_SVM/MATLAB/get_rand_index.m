function new_ind_val = get_rand_index(n_samples, existing_ind_val)
    new_ind_val = existing_ind_val;
    cnt = 0;
    while new_ind_val == existing_ind_val && cnt<1000
        new_ind_val = randi(n_samples);
        cnt = cnt + 1;
    end
end