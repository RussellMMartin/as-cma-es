function f = f_sphere(X)
    X = reshape(X, [], 1);  % Ensure X is a column vector
    f = sum(X .^ 2, 1) + 1;
end
