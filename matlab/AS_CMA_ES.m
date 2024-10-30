%{
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AS_CMA_ES is a CLASS to determine the minimum sample time
such that CMA-ES ordering is still precise. Made for MATLAB R2017b. 
 
PARAMETERS REQUIRED FOR SETUP:

    E_t: E(t). Matrix containing various potential times and the SD of
        errors associated with making a measurement that long. In other words,
        a measurement for time t1 results in an average error of 0 percent and
        an SD error of e1 percent. E(t) contains [[t1, t2, ...]; [e1, e2, ...]]
        E(t) should be experimentally determined prior to optimization. Size =
        (2, any). First row is sample time, second is error (%). The units of
        time (e.g. sec) determine the units used in the rest of the script. The
        smallest and largest times passed here represent the smallest and
        largest sample times (i.e. t_min and t_max) that this algorithm will choose.
    beta: threshold for precision - typically set to 1.3, larger values
        give longer chosen sample times
    y_hat: matrix of expected min (y_hat_min) and (y_hat_min) cost, to be
        used for the first generation's dist vs. deltaCost . Shape = (2,1)
    N: dimensionality of landscape
    verboseType: prints useful information about as-cma-es state variables in
        ask() and tell() functions. 0 = nothing, 1 = print warnings, 
        2 = print chosen sample times + state updates + warnings. 

VARIABLES PASSED DURING USE:

    paramsThisGen: Matrix of parameters to be tested this generation.
        Size = (N,lamba) where N is dimension of space and lambda is conditions
        per generation. This must (!) be normalized such that the minimum allowed
        value is 0 and the maximum allowed value is 1.
    costsThisGen: Matrix of costs associated with parameters tested this
        generation. Size = (lambda, 1)
    nCondition: condition number whose sample time we're trying to determine,
        indexing starting at 1.

METHODS (see above for variable formatting):

    AS_CMA_ES(E_t, beta, y_hat, N, verboseType): called first, used to intitialize priors.
 
    ask_single_sample_time(paramsThisGen, nCondition): Determine the sample time of one
        condition in a generation. Returns a single value. An alternative
        to ask_all_sample_times.

    ask_all_sample_times(paramsThisGen): Determine the sample time of all candidates in
        a generation. Returns a matrix of size = (lambda,1).
 
    tell_generation_results(paramsThisGen, costsThisGen): tell the costs of
        all candidates to the algorithm. This is used to update the
        relationship between parameter distance and cost difference.
 
EXAMPLE USE:

    % 1. initialize once with the constructor function
    as_cma = AS_CMA_ES(E_t, beta, y_hat, N, verboseType);
                
    % 2a. during CMA-ES, can either get length of a single sample time
    sample_time = as_cma.ask_single_sample_time(paramsThisGen, nCondition);

    % 2b. Or, can get the sample times of an entire generation
    sample_times = as_cma.ask_all_sample_times(paramsThisGen);

    % 3. After a generation, tell the measured costs to update the cost per distance relationship
    as_cma = as_cma.tell_generation_results(paramsThisGen, costsThisGen);
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%}
classdef AS_CMA_ES 
    properties 
        E_t                             % E(t) error vs time function
        beta                            % Desired signal-to-noise ratio, suggested beta=1.3
        N                               % Dimensionality of space
        k_avg                           % Avg inter-candidate distance-cost difference
        y_avg                           % Avg cost of the local region of search
        d_max                           % Max possible distance in the (0-1 normalized) space
        verboseType                     % 0 = no prints, 1 = print warnings, 2 = print everything        
    end
    
    methods
        
        function obj = AS_CMA_ES(E_t, beta, y_hat, N, verboseType)
            % AS_CMA_ES     Sets up AD-CMA-ES object with priors.
            %   SYNATX: as_cma = AS_CMA_ES(E_t, beta, y_hat, N, verboseType);
            %   INPUTS: 
            %       - E_t: [matrix size (2,any)] Error vs. time matrix. By default this will be interpolated.
            %       - beta: [number] Desired signal-to-noise ratio, recommended value 1.3.
            %       - y_hat: [matrix size (2,1)] Estimated min and max cost for generation 1.
            %       - N: [number] dimensionality of search space. 
            %       - verboseType [number]: 0 (none), 1 (warnings), or 2 (warnings+other updates) 
            %   OUTPUTS: 
            %       - AS_CMA_ES object setup in accordance with inputs. 
            
            interpolate_E_t = true;
            
            % data validation
            % (1) check E_t
            E_t_validation = size(E_t,1)==2 && isa(E_t, 'double');
            assert(E_t_validation, 'E_t must be matrix size(2,any) and class double')
            % (2) check beta
            beta_validation = isa(beta, 'double') && beta > 0 && beta <= 100;
            assert(beta_validation, 'beta must be a double between 0 and 100 (probably 1.3)')
            % (3) check y_hat
            y_hat_validation = isa(y_hat,'double') && numel(y_hat)==2 && y_hat(1) ~= y_hat(2);
            assert(y_hat_validation, 'y_hat must be a double with 2 elements that are not equal to eachother')
            % (4) check N
            N_validation = isa(N,'double') && numel(N)==1 && N > 1;
            assert(N_validation, 'N must be number > 1');
            % (4) check verboseType
            vb_validation = isa(verboseType,'double') && numel(N)==1 && ismember(verboseType, [0 1 2]);
            assert(vb_validation, 'verboseType must be a number in set {0, 1, 2}');
            
            % if validations passed, save variables
            if interpolate_E_t
                time_interp = linspace(E_t(1, 1), E_t(1, end), 1000);
                err_interp = interp1(E_t(1, :), E_t(2, :), time_interp);
                obj.E_t = [time_interp; err_interp];
            else
                obj.E_t = E_t;
            end
            obj.beta = beta;
            obj.verboseType = verboseType;
            obj.N = N;
            
            % knowing N, we can get d_max, and knowing y_hat, we can initialize k_avg and y_avg
            obj.d_max = norm(ones(obj.N,1));
            obj.k_avg = abs(y_hat(1) - y_hat(2));
            obj.y_avg = mean([y_hat(1), y_hat(2)]);
        end
        
        function sample_time = ask_single_sample_time(obj, paramsThisGen, nCondition)
            % ask_single_sample_time    Uses AS-CMA-ES algorithm to determine sample time for one condition.
            %   SYNATX: sample_time = as_cma.ask_single_sample_time(paramsThisGen, nCondition);
            %   INPUTS: 
            %       - paramsThisGen: [matrix size (N, lambda)]: Candidates to test this generation. Must be 0-1 normalized. 
            %       - nCondition: [number]: Candidate of interest to determine sample time (indexing starting at 1). 
            %   OUTPUTS: 
            %       - t_return: time to sample the candidate for 
            
            % check that paramsThisGen is (N, any) and 0-1 normalized
            paramsThisGen_validation = size(paramsThisGen,1) == obj.N && ...
                all(min(paramsThisGen) >= 0) && ...  
                all(max(paramsThisGen) <= 1);
            assert(paramsThisGen_validation, 'paramsThisGen must be shape (N, lambda) and be normalized to be between 0 and 1.')
            
            % get the distance from this candidate to all others, then normalize
            lambda = size(paramsThisGen, 2);
            distTable = zeros(lambda,1);
            for i = 1:lambda
                distTable(i) = norm(paramsThisGen(:, nCondition) - paramsThisGen(:, i));
                if distTable(i) == 0 && i ~= nCondition && obj.verboseType > 0
                    fprintf('\n WARNING this candidate %i is exactly the same as candidate %i', nCondition, i);
                end
            end
            distTable = distTable ./ obj.d_max;
            
            % get distance of closest condition (that isn't the candidate itself)
            distTable_sorted = sort(distTable);
            distOfClosestCondition = distTable_sorted(2);
            
            % find shortest sample time that satisfies beta given spacing and estimate of local cost landscape
            sdErrors = obj.E_t(2,:);
            d_in = distOfClosestCondition;
            
            epsilon_desired = obj.k_avg * d_in / ( sqrt(2) * obj.y_avg * obj.beta);
            
            if epsilon_desired > obj.E_t(2,1)                       % if a very large error is desired, sample for t_min
                sample_time_idx = 1;
            elseif epsilon_desired < obj.E_t(2,end)                 % if a very small error is desired, sample for t_max
                sample_time_idx = size(obj.E_t,2);
            else                                                    % if a moderate (between the smallest and largest possible errors) error desired
                sign_change = diff(sign(sdErrors - epsilon_desired)); % find all zero crossings of errors - desired_error 
                sample_time_idx = find(sign_change ~= 0, 1);             % find first zero crossing
            end
            
            sample_time = obj.E_t(1, sample_time_idx);
            if obj.verboseType == 2
                fprintf('\n Candidate %i is %.2f from nearest neighbor, dur is %.2f', nCondition, distOfClosestCondition, sample_time);
            end
        end
        
        function sample_times = ask_all_sample_times(obj, paramsThisGen)
            % ask_all_sample_times    Uses AS-CMA-ES algorithm to determine sample time for all conditions in a generation.
            %   SYNATX: sample_times = as_cma.ask_single_sample_time(paramsThisGen, nCondition);
            %   INPUTS: 
            %       - paramsThisGen: [matrix size (N, lambda)] Candidates to test this generation. Must be 0-1 normalized. 
            %   OUTPUTS: 
            %       - sample_times: [matrix size (lambda,1)] time to sample the all candidates for 
            
            % validate that paramsThisGen is size (N, lambda)
            assert(size(paramsThisGen,1) == obj.N, 'paramsThisGen must be shape (N, lambda)')
            
            % initialize return sample times
            lambda = size(paramsThisGen, 2);
            sample_times = zeros(lambda, 1);
            
            % for each condition, determine sample time
            for i = 1:lambda
                sample_times(i) = obj.ask_single_sample_time(paramsThisGen, i);
            end
            
        end
        
        function obj = tell_generation_results(obj, paramsThisGen, costsThisGen)
            
            % tell_generation_results    Updates AS-CMA-ES state variables with information about the most recently evaluated candidates
            %   SYNATX: sample_times = as_cma.ask_single_sample_time(paramsThisGen, nCondition);
            %   INPUTS: 
            %       - paramsThisGen: [matrix size (N, lambda)] Candidates tested this generation. Must be 0-1 normalized. 
            %       - costsThisGen: [matrix size (lambda, 1)] Measured costs of the candidates 
            %   OUTPUTS: 
            %       - AS_CMA_ES object with updated state variables reflecting latest generation's results 
            
            validation = size(paramsThisGen, 1)==obj.N && ...       % check size of param array
                size(paramsThisGen,2) == numel(costsThisGen) && ... % check that num params and num costs are equal
                all(min(paramsThisGen) >= 0) && ...                 % check that params are 0-1 scaled
                all(max(paramsThisGen) <= 1);
            assert(validation, 'tell_generation_results: param array shape incorrect or not 0-1 normalized')
            if ~validation
                table(paramsThisGen)
                table(costsThisGen)
            end
           
            nConds = size(paramsThisGen, 2);
            allCostsDiffs = zeros(nConds^2,1);
            allParamDists = zeros(nConds^2,1);
            counter = 1;
            
            % compare each condition to each other condition to find the
            % avg relationship between {nearest neighbor distance} and {nearest
            % neighbor cost difference}
            for i=1:nConds
                for j=1:nConds
                    dist_diff = norm(paramsThisGen(:,i) - paramsThisGen(:, j)) / obj.d_max;
                    cost_diff = abs(costsThisGen(i) - costsThisGen(j));
                    allCostsDiffs(counter) = cost_diff;
                    allParamDists(counter) = dist_diff;
                    counter = counter + 1;
                end
            end

            % fit distance-difference to linear function
            fitfun = fittype( @(m,x) m*x+0);
            [fitted_curve,~] = fit(allParamDists, allCostsDiffs, fitfun, 'StartPoint', 1);
            
            % save results to obj
            obj.k_avg = fitted_curve.m;
            obj.y_avg = mean(costsThisGen);
            
            if obj.verboseType == 2
                fprintf('\n k_avg = %.2f , y_avg = %.2f ', obj.k_avg, obj.y_avg)
                fprintf('\n params: ');
                fprintf(mat2str(paramsThisGen))
                fprintf('\n costs: ');
                fprintf(mat2str(costsThisGen))
            end
        end
    end
end

