from simQ_base import model_constructor, run_custom_sim, run_sim_pipeline
import pandas as pd
import numpy as np


DEFAULT_MMn_ARGS = dict(arrival_rate_0 = 10,
                        process_coeff = 1,
                        work_period = 60*24)

DEFAULT_GGn_ARGS = dict(service_min=0)  




###############
#Cost Functions
###############

def cost_func_quad_3(mus,metric,args):

    '''  
    Returns the cost attribted to a given policy (\mu_1,\mu_2) and the resultant
    runtime of the second queue. Specific cost function

    Parameters:
    -----------
    mus: list, float
        The decision variables for the service rate

    metric: float,int
        The resultant metric from running the simulation
    
    args: list,float
        A list of arguments for the cost function i.e. c_1, c_2, c_3
 
    '''
    mu_1 = mus[0]
    mu_2 = mus[1]

    c_1 = args[0]
    c_2 = args[1]
    c_3 = args[2]

    return c_1*mu_1**2 + c_2*mu_2**2 + c_3 * (metric/60)**(1/2)

DEFAULT_CF = cost_func_quad_3

def metric_expon(mus,metric,args):

    '''  
    Takes the specified exponent of the stated metric.

       Parameters:
    -----------
    mus: list, float
        The decision variables for the service rate (ignored here)

    metric: float,int
        The resultant metric from running the simulation
    
    args: float
        The exponent value 
 
    '''
    expon = args

    return (metric/60)**(expon)
##########
#LF Models
##########

def B_metric_Q1_LF_QnD(mu_1,lmbda = 10,T = 24,eps = 0.05,mu_shft = 1.3):
    '''
        A quick and dirty approximation for an MM1 queue which ouputs 
        the backlog for a given lambda and mu. While 
        lambda < mu_1, steady state analytical approximation is used
        for L_q.

        While lambda > mu_1, we take the difference and multiply by T.

        Paramaters:
        -----------
        mu: float
            the name of the file which we are interpolating
        lmbda: float
            The arrival rater for the queue.
        T: float
            The service period of Queue 1
        eps: float
            epsilon, a sufficiently small quantity which moves the funciton
            for which \rho < 1 outside of the region where values for queue
            length explode to infinity (tunable paramater)
        mu_shft: float
            A term which translates the function for which \rho > 1 along the
            mu axis, so as to crudely join the two functions


    '''

    B = 0
    rho = lmbda/mu_1

    if rho < 1 - eps:

        B = rho/(1-rho) - rho 
    
    else:

        B = (lmbda - mu_1 + mu_shft)*T/60 

    return int(np.round(B))   

def R_metric_Q2_LF(L_T,mu_2):
    '''
    Given a measurment of the final queue length L_t for Queue 1
    and a value for service rate in Queue 2, this function outputs
    an estimate for the runtime of Queue 2, denoted as R_bar(L_T,mu_2)

    Inputs
    ------
    L_T: int
        The number of people left in Queue 2 (backlog) after service period T
    
    mu_2: float
        The service rate of Queue 2

    Output
    -----
    R: float, units: mins
        The estimated runtime for the provided inputs

    '''

    return L_T * 60/mu_2



######
#Utils
######

class cf_primer:

    def __init__(self, cf_args, cost_function = DEFAULT_CF):

        self.args = cf_args
        self.cf = cost_function

    def eval(self,mus,metric):

        return self.cf(mus,metric,self.args)






################
#Data Collection
################

def run_M_snobs_test(M, service_rates, seed, cost_function, queue_1_parms = dict(), queue_2_parms = dict()):
    '''
    Obtains a the average and variance of thecost function value from the sample average of 
    M observations of the queue network simulation for a given scenario and cost function.

    The model used by this function is a bi node queueing simulation network. 
    Where customers arrive at queue 1 for service period T, are serverd.  Any remaining
    unserved customers after T are sent to queue 2. Queue 2 runs until this backlog is cleared.

    IMPORTANT:
    ----------
    This is a test function which compares two methods of calculating the cost function using the expected output
    from M simulation observations:
    1. E(cost_function)
    2. cost_function = args + E(sim_obvs)
    
    Params:
    ------
    M: int
        Number of observations of queue network
    
    service_rates: List 1x2
        A vector containing the policy (chosen service rates) for this run of the simulation
    
    seed: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model to collect results
    
    cost_function: cf_primer class
        The initialised cost function given a function and specified arguments
    

    queue_1_parms: dict, optional (default=dict() (empty dictionary))
        Queue 1's specified configuration for the experiment as a dicitonary.
        See documentation for correct format. 

        Optional: If no argument is passed, DES uses system defaults (see notes)

    queue_2_parms: dict, optional (default=dict() (empty dictionary))
        Ditto for Queue 2
        
    Returns:
    --------
    results_cst1: pandas DataFrame
        The average cost and sample variance calculated from the sample average of each of the M simulations
        for cost calculation method 1
    results_cst2: pandas DataFrame
        The average cost and sample variance calculated from the sample average of each of the M simulations
        for cost calculation method 2
    results_rt: pandas DataFrame
        The average runtime and its sample variance for queue 1; 
        calculated from the sample average from queue 1. 
    '''

    all_q2_csts = []
    all_q2_runtimes = []


    queue_1 = model_constructor('MMn',queue_1_parms)
    queue_2 = model_constructor('GGn',queue_2_parms)

    for rep in range(M):

        runtime = run_sim_pipeline(service_rates, seed, queue_1, queue_2)

        cost = cost_function(service_rates,runtime)

        all_q2_csts.append(cost)
        all_q2_runtimes.append(runtime)

        seed += 1
    
    
    cost_1 = np.array(all_q2_csts).mean()
    var_1 = np.array(all_q2_csts).var()

    mean_rt = np.array(all_q2_runtimes).mean()
    var_rt = np.array(all_q2_runtimes).var()

    cost_2 = cost_function(service_rates,mean_rt)
    
    var_2 = (cost_2*var_rt/(2*mean_rt))**2 #Biased estimator of cost_2 variance

    



    #format and return results in a dataframe
    result_cst1 = pd.DataFrame(dict(mu_1 = service_rates[0], 
                               mu_2 = service_rates[1],
                               mean_cst = cost_1,
                               var_cst = var_1), index=[0])

    result_cst2= pd.DataFrame(dict(mu_1 = service_rates[0], 
                              mu_2 = service_rates[1],
                              mean_cst = cost_2,
                              var_cst = var_2), index=[0])
    
    result_rt = pd.DataFrame(dict(mu_1 = service_rates[0], 
                             mu_2 = service_rates[1],
                             mean_rt = mean_rt,
                             var_rt = var_rt), index=[0])

    return result_cst1, result_cst2, result_rt


def run_M_snobs_default(M, service_rates, seed, cost_function, queue_1_parms = dict(), queue_2_parms = dict(),log10_out=False):
    '''
    Obtains a the average and variance of thecost function value from the sample average of 
    M observations of the queue network simulation for a given scenario and cost function.

    The model used by this function is a bi node queueing simulation network. 
    Where customers arrive at queue 1 for service period T, are serverd.  Any remaining
    unserved customers after T are sent to queue 2. Queue 2 runs until this backlog is cleared
    
    Params:
    ------
    M: int
        Number of observations of queue network
    
    service_rates: List 1x2
        A vector containing the policy (chosen service rates) for this run of the simulation
    
    seed: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model to collect results
    
    cost_function: cf_primer class
        The initialised cost function given a function and specified arguments
    

    queue_1_parms: dict, optional (default=dict() (empty dictionary))
        Queue 1's specified configuration for the experiment as a dicitonary.
        See documentation for correct format. 

        Optional: If no argument is passed, DES uses system defaults (see notes)

    queue_2_parms: dict, optional (default=dict() (empty dictionary))
        Ditto for Queue 2

    log10_out: bool, optional (default = False)
        Directly transforms the simulation output by log10, for ease of numerical stability during the BO procedure.
        
    Returns:
    --------
    seed: int
        Passes the sample seed after M runs
    cost_out: float
        The average cost calculated from the sample average of each of the M simulations.
        Note: cost is calculated using method 1 (see notes)
    var_out: float
        The sample variance of the average cost
    '''

    all_q2_csts = []


    queue_1 = model_constructor('MMn',queue_1_parms)
    queue_2 = model_constructor('GGn',queue_2_parms)

    for rep in range(M):

        runtime = run_sim_pipeline(service_rates, seed, queue_1, queue_2)
        #In the future I will allow for this part of the code to be swappable for different configurations of simulation

        cost = cost_function.eval(service_rates,runtime)

        if log10_out: #applys transform if specified
            
            cost = np.log10(cost)

        all_q2_csts.append(cost)
        seed += 1
    

    
    cost_out = np.array(all_q2_csts).mean()
    var_out = np.array(all_q2_csts).var()


    return seed, cost_out, var_out


def run_M_snobs_LF_HF(M, service_rates, seed, cost_function, queue_1_parms = dict(), queue_2_parms = dict()):
    '''
    Obtains a the average and variance of thecost function value from the sample average of 
    M observations of the queue network simulation for a given scenario and cost function.

    The model used by this function is a bi node queueing simulation network. 
    Where customers arrive at queue 1 for service period T, are serverd.  Any remaining
    unserved customers after T are sent to queue 2. Queue 2 runs until this backlog is cleared
    
    Params:
    ------
    M: int
        Number of observations of queue network
    
    service_rates: List 1x2
        A vector containing the policy (chosen service rates) for this run of the simulation
    
    seed: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model to collect results
    
    cost_function: cf_primer class
        The initialised cost function given a function and specified arguments
    

    queue_1_parms: dict, optional (default=dict() (empty dictionary))
        Queue 1's specified configuration for the experiment as a dicitonary.
        See documentation for correct format. 

        Optional: If no argument is passed, DES uses system defaults (see notes)

    queue_2_parms: dict, optional (default=dict() (empty dictionary))
        Ditto for Queue 2
        
    Returns:
    --------
    seed: int
        Passes the sample seed after M runs
    cost_out: float
        The average cost calculated from the sample average of each of the M simulations.
        Note: cost is calculated using method 1 (see notes)
    var_out: float
        The sample variance of the average cost
    '''

    all_q2_csts = []

    queue_2 = model_constructor('GGn',queue_2_parms)

    B = B_metric_Q1_LF_QnD(service_rates[0],
                           lmbda=queue_1_parms['arrival_rate_0'],
                           T = queue_1_parms['work_period'])

    for rep in range(M):

        runtime = run_custom_sim(service_rates[1], seed, queue_2, input =B)
        #In the future I will allow for this part of the code to be swappable for different configurations of simulation

        cost = cost_function.eval(service_rates,runtime)

        all_q2_csts.append(cost)
        seed += 1
    
    
    cost_out = np.array(all_q2_csts).mean()
    var_out = np.array(all_q2_csts).var()


    return seed, cost_out, var_out


def run_M_snobs_HF_LF(M, service_rates, seed, cost_function, queue_1_parms = dict(), queue_2_parms = dict()):
    '''
    Obtains a the average and variance of thecost function value from the sample average of 
    M observations of the queue network simulation for a given scenario and cost function.

    The model used by this function is a bi node queueing simulation network. 
    Where customers arrive at queue 1 for service period T, are serverd.  Any remaining
    unserved customers after T are sent to queue 2. Queue 2 runs until this backlog is cleared
    
    Params:
    ------
    M: int
        Number of observations of queue network
    
    service_rates: List 1x2
        A vector containing the policy (chosen service rates) for this run of the simulation
    
    seed: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model to collect results
    
    cost_function: cf_primer class
        The initialised cost function given a function and specified arguments
    

    queue_1_parms: dict, optional (default=dict() (empty dictionary))
        Queue 1's specified configuration for the experiment as a dicitonary.
        See documentation for correct format. 

        Optional: If no argument is passed, DES uses system defaults (see notes)

    queue_2_parms: dict, optional (default=dict() (empty dictionary))
        Ditto for Queue 2
        
    Returns:
    --------
    seed: int
        Passes the sample seed after M runs
    cost_out: float
        The average cost calculated from the sample average of each of the M simulations.
        Note: cost is calculated using method 1 (see notes)
    var_out: float
        The sample variance of the average cost
    '''

    all_q2_csts = []

    queue_1 = model_constructor('MMn',queue_1_parms)

    for rep in range(M):

        B = run_custom_sim(service_rates[0], seed, queue_1)
        #In the future I will allow for this part of the code to be swappable for different configurations of simulation

        runtime = R_metric_Q2_LF(B,service_rates[1])
        cost = cost_function.eval(service_rates,runtime)

        all_q2_csts.append(cost)
        seed += 1
    
    
    cost_out = np.array(all_q2_csts).mean()
    var_out = np.array(all_q2_csts).var()


    return seed, cost_out, var_out



def run_ob_LF_LF(M, service_rates,seed,cost_function, queue_1_parms = dict(), queue_2_parms = dict()):
    '''
    Obtains a LF observation of the queueing network case used in our expeirments 
    
    Params:
    ------
    M int:
        Number of repeat observations under service rate

    service_rates: List 1x2
        A vector containing the policy (chosen service rates) for this run of the simulation.
    
    seed: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        not used, but maintained inclusion for ease of integration with outher funcitons
    
    cost_function: cf_primer class
        The initialised cost function given a function and specified arguments

    queue_1_parms: dict, optional (default=dict() (empty dictionary))
        Queue 1's specified configuration for the experiment as a dicitonary.
        See documentation for correct format. 

        Optional: If no argument is passed, DES uses system defaults (see notes)

    queue_2_parms: dict, optional (default=dict() (empty dictionary))
        Ditto for Queue 2
        
    Returns:
    --------
    seed: int
        Passes the sample seed after M runs
    cost_out: float
        The average cost calculated from the sample average of each of the M simulations.
        Note: cost is calculated using method 1 (see notes)
    var_out: float
        The sample variance of the average cost
    '''


    B = B_metric_Q1_LF_QnD(service_rates[0],
                           lmbda=queue_1_parms['arrival_rate_0'],
                           T = queue_1_parms['work_period'])

    runtime = R_metric_Q2_LF(B,service_rates[1])

    cost = cost_function.eval(service_rates,runtime)

    return seed, cost, 0 
