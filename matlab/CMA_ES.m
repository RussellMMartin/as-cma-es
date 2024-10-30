classdef CMA_ES
    % R. M. Martin and S. H. Collins 2024
    % Barebones ask-and-tell CMA-ES interface
    % Adapted from Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv preprint arXiv:1604.00772.
    
    properties
        N             % Number of objective variables/problem dimension
        mins          % min allowable value for each of N dimensions
        maxs          % max allowable value for each of N dimensions
        lambda        % Population size, offspring number
        mu            % Number of parents/points for recombination
        weights       % Weights for recombination
        mueff         % Variance-effectiveness of sum w_i x_i
        cc            % Time constant for cumulation for C
        cs            % Time constant for cumulation for sigma control
        c1            % Learning rate for rank-one update of C
        cmu           % Learning rate for rank-mu update
        damps         % Damping for sigma
        sigma         % Coordinate-wise standard deviation (step size)
        pc            % Evolution path for C
        ps            % Evolution path for sigma
        B             % Coordinate system matrix
        D             % Diagonal D defines the scaling
        C             % Covariance matrix
        invsqrtC      % C^-1/2
        xmean_01      % Mean of the population, used for CMA math
        xmean_real    % Mean of the population, in real-world terms
        counteval     % Counter for evaluations
        eigeneval     % Track update of B and D
        chiN          % Expectation of ||N(0,I)||
    end
    
    methods
        function obj = CMA_ES(N, sigma, mean, mins, maxs)
            % Constructor to initialize the CMA-ES parameters
            obj.N = N;
            obj.mins = mins;
            obj.maxs = maxs;
            obj.sigma = sigma;
            obj.lambda = 4 + floor(3 * log(N));  % Population size
            obj.mu = floor(obj.lambda / 2);       % Number of parents
            obj.weights = log(obj.mu + 1/2) - log(1:obj.mu)'; % Weights
            obj.weights = obj.weights / sum(obj.weights); % Normalize
            obj.mueff = sum(obj.weights)^2 / sum(obj.weights.^2); % mueff
            
            % Strategy parameters
            obj.cc = (4 + obj.mueff / N) / (N + 4 + 2 * obj.mueff / N);
            obj.cs = (obj.mueff + 2) / (N + obj.mueff + 5);
            obj.c1 = 2 / ((N + 1.3)^2 + obj.mueff);
            obj.cmu = min(1 - obj.c1, 2 * (obj.mueff - 2 + 1 / obj.mueff) / ((N + 2)^2 + obj.mueff));
            obj.damps = 1 + 2 * max(0, sqrt((obj.mueff - 1) / (N + 1)) - 1) + obj.cs;
            
            % Internal strategy parameters
            obj.pc = zeros(N, 1);
            obj.ps = zeros(N, 1);
            obj.B = eye(N, N);
            obj.D = ones(N, 1);
            obj.C = obj.B * diag(obj.D.^2) * obj.B';
            obj.invsqrtC = obj.B * diag(obj.D.^-1) * obj.B';
            obj.eigeneval = 0;
            obj.chiN = N^0.5 * (1 - 1/(4*N) + 1/(21*N^2));
            obj.counteval = 0;
            obj.xmean_real = mean;
            obj.xmean_01 = (mean - mins) ./ (maxs - mins);
            obj.xmean_01 = min(max(obj.xmean_01, 0), 1);
        end
        
        function [candidates, candidates_01] = ask(obj)
            % Generate new candidates
            % Note that this does not sample the mean
            candidates = zeros(obj.N, obj.lambda);          % candidates scaled to real limits
            candidates_01 = zeros(obj.N, obj.lambda);       % candidates scaled to be between 0-1
            for k = 1:obj.lambda
                candidate_01 = obj.xmean_01 + obj.sigma * obj.B * (obj.D .* randn(obj.N, 1));
                candidate_01 = max(min(candidate_01, 1), 0);
                
                candidates_01(:,k) = candidate_01;
                candidates(:, k) = candidate_01 .* (obj.maxs - obj.mins) + obj.mins;
            end
        end
        
        function obj = tell(obj, candidates, fitness)
            % scale input candidates back to 0-1
            candidates_01 = zeros(size(candidates));
            for k = 1:size(candidates,2)
                candidates_01(:, k) = (candidates(:,k) - obj.mins) ./ (obj.maxs - obj.mins);
            end
            
            % Update CMA-ES based on the fitness of candidates
            [~, arindex] = sort(fitness); % Sort by fitness
            xold_01 = obj.xmean_01;
            obj.xmean_01 = candidates_01(:, arindex(1:obj.mu)) * obj.weights;   % Recombination
            obj.xmean_01 = min(max(obj.xmean_01, 0), 1);
            obj.xmean_real = obj.xmean_01 .* (obj.maxs - obj.mins) + obj.mins;
            
            % Cumulation: Update evolution paths
            obj.ps = (1-obj.cs)*obj.ps ...
                + sqrt(obj.cs*(2-obj.cs)*obj.mueff) * obj.invsqrtC * (obj.xmean_01-xold_01) / obj.sigma;
            hsig = norm(obj.ps)/sqrt(1-(1-obj.cs)^(2*obj.counteval/obj.lambda))/obj.chiN < 1.4 + 2/(obj.N+1);
            obj.pc = (1-obj.cc)*obj.pc ...
                + hsig * sqrt(obj.cc*(2-obj.cc)*obj.mueff) * (obj.xmean_01-xold_01) / obj.sigma;
            
            % Adapt covariance matrix C
            artmp = (1/obj.sigma) * (candidates_01(:,arindex(1:obj.mu))-repmat(xold_01,1,obj.mu));
            obj.C = (1-obj.c1-obj.cmu) * obj.C ...                  % regard old matrix
                + obj.c1 * (obj.pc*obj.pc' ...                 % plus rank one update
                + (1-hsig) * obj.cc*(2-obj.cc) * obj.C) ... % minor correction if hsig==0
                + obj.cmu * artmp * diag(obj.weights) * artmp'; % plus rank mu update
            
            % Adapt step size sigma
            obj.sigma = obj.sigma * exp((obj.cs/obj.damps)*(norm(obj.ps)/obj.chiN - 1));
            
            % Decomposition of C into B*diag(D.^2)*B' (diagonalization)
            if obj.counteval - obj.eigeneval > obj.lambda/(obj.c1+obj.cmu)/obj.N/10  % to achieve O(N^2)
                obj.eigeneval = obj.counteval;
                obj.C = triu(obj.C) + triu(obj.C,1)'; % enforce symmetry
                [obj.B,obj.D] = eig(obj.C);           % eigen decomposition, B==normalized eigenvectors
                obj.D = sqrt(diag(obj.D));        % D is a vector of standard deviations now
                obj.invsqrtC = obj.B * diag(obj.D.^-1) * obj.B';
            end
            
            obj.counteval = obj.counteval + obj.lambda; % Update evaluation count
        end
    end
end