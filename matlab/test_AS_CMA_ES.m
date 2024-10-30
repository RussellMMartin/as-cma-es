% test_AS_CMA_ES.m is a script to demonstrate the implementation of
% AS-CMA-ES in a single optimization run. Files/functions used here:
%   - AS_CMA_ES.m: The AS-CMA-ES library. 
%   - E(t).csv: Contains measurement times (1st row) and expected % errors (2nd row) 
%   - CMA_ES.m: The CMA-ES optimization algorithm, adapted to an ask-and-tell style for this demo. 
%   - f_sphere.m: The ground-truth cost function we're optimizing in this demo. 
% R. M. Martin and S. H. Collins 2024

% 1. Initialize CMA-ES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rng(1)                                      % seed, for repeatable results
N = 20;                                     % dimensionality
sigma = 0.3;                                % initial CMA search size
mins = ones(N,1)*-10;                       % min parameter in search space
maxs = ones(N,1)*10;                        % max parameter in search space
mean = ones(N,1)*5;                         % starting search point
cma = CMA_ES(N, sigma, mean, mins, maxs);   % create cma object using above settings

% 2. Initialize AS-CMA-ES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
use_adaptive_sampling = 1;                  % boolean to use adaptive or static sampling
E_t = csvread('E(t).csv');                  % read in error vs time file
if use_adaptive_sampling
    beta = 1.3;                             % desired signal-to-noise ratio, recommend 1.3
    y_hat = [1; 500];                       % estimated min/max possible cost that could be found in gen 1
    verboseType = 0;                        % 0 turns off all AS-CMA prints
    as_cma = AS_CMA_ES(E_t, beta, y_hat, N, verboseType); % create as_cma object using above settings
else
    static_sample_time = 0.5;               % if not using AS-CMA, this sets the time of all samples. 
                                            %   Generally, short static sample times converge to poor solutions quickly, 
                                            %   while long static sample times converge to quality solutions slowly
end

% 3. Setup optimization loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
t_max = 5000;                               % amount of time to optimize
t = 0;                                      % variable to track current optimization time
track_mean_cost = f_sphere(cma.xmean_real); % tracks mean cost over time
track_mean_param_value = cma.xmean_real;    % tracks mean parameter values over time
track_gen_time = 0;                         % tracks generation start time
track_selected_sample_times = [];           % tracks the chosen sample times (by AS-CMA-ES or the static_sample_time)

% 4. Do optimization loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
while t < t_max
    % 4a. Get candidates to evaluate 
    [candidates, candidates_01] = cma.ask();   
    
    %4b. Determine how long to sample each candidate
    if use_adaptive_sampling
        sample_times = as_cma.ask_all_sample_times(candidates_01);
    else
        sample_times = ones(cma.lambda,1) * static_sample_time;
    end
    
    % 4c. Evaluate candidates, adding noise based on sample time
    costs_noisy = zeros(cma.lambda,1);
    for c=1:cma.lambda                          
        cost_true = f_sphere(candidates(:,c));  
        noise_level = interp1(E_t(1, :), E_t(2, :), sample_times(c));
        noise_multiplier = max((1 + randn * noise_level), 0);
        cost_noisy = cost_true * noise_multiplier;
        costs_noisy(c) = cost_noisy;
    end
    
    % 4d. Provide costs back to CMA-ES and AS-CMA-ES for state updating
    cma = cma.tell(candidates, costs_noisy);
    if use_adaptive_sampling
        as_cma = as_cma.tell_generation_results(candidates_01, costs_noisy);
    end
    
    % 4e. Update optimization trackers
    t = t + sum(sample_times);
    track_mean_cost = [track_mean_cost; f_sphere(cma.xmean_real)];
    track_mean_param_value = [track_mean_param_value cma.xmean_real];
    track_gen_time = [track_gen_time; t];
    track_selected_sample_times = [track_selected_sample_times; sample_times]; 
end

% 5. Evaluate results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
figure(); 
subplot(2,3,1);
plot(track_gen_time, track_mean_cost, 'Marker', '.'); hold on;
plot([0 t_max], [1 1], 'r--'); % true minimum
xlim([0, t_max]);
set(gca, 'YScale', 'log');
title('True mean cost vs time');

subplot(2,3,2);
plot(cumsum(track_selected_sample_times), track_selected_sample_times, 'Marker', '.'); 
xlim([0, t_max]); ylim([min(E_t(1,:)), max(E_t(1,:))]);
title('Selected sample time vs time');

for n = 1:4
    subplot(2,3,n+2); 
    plot(track_gen_time, track_mean_param_value(n,:), 'Marker', '.'); hold on;
    plot([0 t_max], [0 0], 'r--'); % true optimal parameter value
    xlim([0, t_max]);
    title(['Value of param ' num2str(n) ' vs time']);
end