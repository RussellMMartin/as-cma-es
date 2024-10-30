import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
"""
AS_CMA_ES is a CLASS to determine the minimum sample time
such that CMA-ES ordering is still precise. Made for MATLAB R2017b. 
 
PARAMETERS REQUIRED FOR SETUP:

    E_t: E(t). Matrix containing various potential times and the SD of
        errors associated with making a measurement that long. In other words,
        a measurement for time t1 results in an average error of 0 percent and
        an SD error of e1 percent. E(t) contains [[t1, t2, ...]; [e1, e2, ...]]
        E(t) should be experimentally determined prior to optimization. Shape =
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
        Shape = (N,lamba) where N is dimension of space and lambda is conditions
        per generation. This must (!) be normalized such that the minimum allowed
        value is 0 and the maximum allowed value is 1.
    costsThisGen: Matrix of costs associated with parameters tested this
        generation. Shape = (lambda,)
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
    as_cma.tell_generation_results(paramsThisGen, costsThisGen);
"""
class AS_CMA_ES:
    def __init__(self, E_t, beta, y_hat, N, verboseType):
        """
        AS_CMA_ES: Sets up AS-CMA-ES object with priors.
        
        Parameters:
            - E_t: numpy array of shape (2, m). Error (2nd row) vs. time (1st row) matrix.
            - beta: float. Desired signal-to-noise ratio, recommended value 1.3.
            - y_hat: numpy array of shape (2, 1). Estimated min and max cost for generation 1.
            - N: int. Dimensionality of search space.
            - verboseType: int. 0 (none), 1 (warnings), or 2 (warnings+other updates).
        """
        self.verboseType = verboseType
        interpolate_E_t = True
        
        # Data validation
        assert E_t.shape[0] == 2, "E_t must be shape (2, m)"
        assert isinstance(beta, float) and 0 < beta <= 100, "beta must be a float between 0 and 100 (probably 1.3)"
        assert y_hat.shape == (2,) and y_hat[0] != y_hat[1], "y_hat must be shape (2,) with distinct values"
        assert isinstance(N, int) and N > 1, "N must be an integer > 1"
        assert verboseType in [0, 1, 2], "verboseType must be in {0, 1, 2}"
        
        # Interpolation of E_t
        if interpolate_E_t:
            time_interp = np.linspace(E_t[0, 0], E_t[0, -1], 1000)
            err_interp = interp1d(E_t[0, :], E_t[1, :], kind='linear')(time_interp)
            self.E_t = np.vstack((time_interp, err_interp))
        else:
            self.E_t = E_t
        
        # Parameter initialization
        self.beta = beta
        self.N = N
        self.d_max = np.linalg.norm(np.ones((self.N, 1)))
        self.k_avg = abs(y_hat[0] - y_hat[1])
        self.y_avg = np.mean(y_hat)

    def ask_single_sample_time(self, paramsThisGen, nCondition):
        """
        Determine sample time for one condition.
        
        Parameters:
            - paramsThisGen: numpy array of shape (N, lambda). Candidates for the generation.
            - nCondition: int. Index of candidate to determine sample time for.
        
        Returns:
            - float: sample time.
        """
        assert paramsThisGen.shape[0] == self.N, "paramsThisGen must be shape (N, lambda)"
        assert np.all((0 <= paramsThisGen) & (paramsThisGen <= 1)), "paramsThisGen must be 0-1 normalized"
        
        lambda_ = paramsThisGen.shape[1]
        distTable = np.zeros(lambda_)
        
        for i in range(lambda_):
            distTable[i] = np.linalg.norm(paramsThisGen[:, nCondition] - paramsThisGen[:, i])
            if distTable[i] == 0 and i != nCondition and self.verboseType > 0:
                print(f"\n WARNING: candidate {nCondition} is exactly the same as candidate {i}")
        
        distTable /= self.d_max
        distOfClosestCondition = np.partition(distTable, 1)[1]
        
        sdErrors = self.E_t[1, :]
        d_in = distOfClosestCondition
        epsilon_desired = self.k_avg * d_in / (np.sqrt(2) * self.y_avg * self.beta)
        
        if epsilon_desired > sdErrors[0]:
            sample_time_idx = 0
        elif epsilon_desired < sdErrors[-1]:
            sample_time_idx = -1
        else:
            sign_change = np.diff(np.sign(sdErrors - epsilon_desired))
            sample_time_idx = np.where(sign_change != 0)[0][0] + 1
        
        sample_time = self.E_t[0, sample_time_idx]
        
        if self.verboseType == 2:
            print(f"\n Candidate {nCondition} is {distOfClosestCondition:.2f} from nearest neighbor, duration is {sample_time:.2f}")
        
        return sample_time

    def ask_all_sample_times(self, paramsThisGen):
        """
        Determine sample times for all conditions in a generation.
        
        Parameters:
            - paramsThisGen: numpy array of shape (N, lambda). Candidates for the generation.
        
        Returns:
            - numpy array of shape (lambda, 1): Sample times for all candidates.
        """
        assert paramsThisGen.shape[0] == self.N, "paramsThisGen must be shape (N, lambda)"
        
        lambda_ = paramsThisGen.shape[1]
        sample_times = np.zeros(lambda_)
        
        for i in range(lambda_):
            sample_times[i] = self.ask_single_sample_time(paramsThisGen, i)
        
        return np.squeeze(sample_times)

    def tell_generation_results(self, paramsThisGen, costsThisGen):
        """
        Update state variables with the latest generation's results.
        
        Parameters:
            - paramsThisGen: numpy array of shape (N, lambda). Candidates tested this generation.
            - costsThisGen: numpy array of shape (lambda, 1). Measured costs of the candidates.
        """
        assert paramsThisGen.shape[0] == self.N and paramsThisGen.shape[1] == costsThisGen.size, \
            "Mismatch in paramsThisGen shape or costsThisGen size"
        assert np.all((0 <= paramsThisGen) & (paramsThisGen <= 1)), "paramsThisGen must be 0-1 normalized"
        
        nConds = paramsThisGen.shape[1]
        allCostsDiffs = np.zeros(nConds**2)
        allParamDists = np.zeros(nConds**2)
        counter = 0
        
        for i in range(nConds):
            for j in range(nConds):
                dist_diff = np.linalg.norm(paramsThisGen[:, i] - paramsThisGen[:, j]) / self.d_max
                cost_diff = abs(costsThisGen[i] - costsThisGen[j])
                allCostsDiffs[counter] = cost_diff
                allParamDists[counter] = dist_diff
                counter += 1
        
        # Fit distance-difference to linear function
        def fit_function(x, m):
            return m * x
        
        fitted_params, _ = curve_fit(fit_function, allParamDists, allCostsDiffs, p0=[1])
        self.k_avg = fitted_params[0]
        self.y_avg = np.mean(costsThisGen)

        # Verbose output
        if self.verboseType == 2:
            print(f'\n k_avg = {self.k_avg:.2f} , y_avg = {self.y_avg:.2f}')
            print('\n params: ')
            print(paramsThisGen)
            print('\n costs: ')
            print(costsThisGen)
