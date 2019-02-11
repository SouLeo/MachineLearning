function acc = accuracy(empirical, theoretical)
    num_of_tests = length(empirical);
    incorrect_guesses = nnz(empirical - theoretical);
    acc = (num_of_tests-incorrect_guesses)/num_of_tests;
end