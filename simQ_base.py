import simpy
import numpy as np
import itertools

from functools import partial

#############
#Sim Settings
#############

# default random number SET
DEFAULT_RNG_SET = 12345
N_STREAMS = 20 #Number of seperate random number streams to be used for queue subsystems

#############
#Model Parameters
#############

# default results collection period
DEFAULT_WORK_PERIOD =  24*60

DEFAULT_SERVERS = 1

DEFAULT_ARRIVAL_RATE = 10 #Stationary arrival rate (customers/hour)

#For M/G/c queue
DEFAULT_SERVICE_VAR = 2.0 #Normal service times variance (mins)
DEFAULT_SERVICE_MIN = 0.5 #Sampled values > 30 seconds 

#Process Coefficient A
#A \in [0,1] - 1 non-stationary, 0 stationary, 0.5 light non-stationary
DEFAULT_PROC_COEF = 0

#Model Type
DEFAULT_SERVICE_TYPE = 1 #Sets model type to markovian services


DEFAULT_MODEL_PARAMS = dict()


######
#DISTRIBUTIONS
######
class Exponential:
    '''
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, mean, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the exponential distribution
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rng = np.random.default_rng(seed=random_seed)
        self.mean = mean
        
    def sample(self, size=None):
        '''
        Generate a sample from the exponential distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rng.exponential(self.mean, size=size)
    

class Normal:
    '''
    Convenience class for the normal distribution.
    packages up distribution parameters, seed and random generator.

    Use the minimum parameter to truncate the distribution
    '''
    def __init__(self, mean, sigma, minimum=None, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        mean: float
            The mean of the normal distribution
            
        sigma: float
            The stdev of the normal distribution
            
        minimum: float, optional (default=None)
            Used to truncate the distribution (e.g. to 0.0 or 0.5)
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rng = np.random.default_rng(seed=random_seed)
        self.mean = mean
        self.sigma = sigma
        self.minimum = minimum
        
    def sample(self, size=None):
        '''
        Generate a sample from the normal distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        samples = self.rng.normal(self.mean, self.sigma, size=size)

        if self.minimum is None:
            return samples
        elif size is None:
            return max(self.minimum, samples)
        else:
            # index of samples with negative value
            neg_idx = np.where(samples < 0)[0]
            samples[neg_idx] = self.minimum
            return samples
        
        
class Uniform():
    '''
    Convenience class for the Uniform distribution.
    packages up distribution parameters, seed and random generator.
    '''
    def __init__(self, low, high, random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        low: float
            lower range of the uniform
            
        high: float
            upper range of the uniform
        
        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        '''
        self.rand = np.random.default_rng(seed=random_seed)
        self.low = low
        self.high = high
        
    def sample(self, size=None):
        '''
        Generate a sample from the uniform distribution
        
        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        '''
        return self.rand.uniform(low=self.low, high=self.high, size=size)      

        
######
#Utils
######

def trace(msg, TRACE = False):
    '''
    Utility function for printing output as the
    simulation model executes.
    Set the TRACE constant to False, to turn tracing off.
    
    Params:
    -------
    msg: str
        string to print to screen.
    '''
    if TRACE:
        print(msg)

def lambda_func(t,T,lmbda_0,A):
    '''
    A funciton for the change in arrival rate
    lambda(t). When A=0, we have a stationary
    arrival rate and lambda(t) = lmbda_0.
    
    Params:
    -------
    t: float
        Current simulation clock time
    T: float
        The runtime period of the simulation
    lmbda_0: float
        The baseline value of lambda
    A: float
        The stationary/non-stationary coefficient.
        A=0 stationary, A > 0 non-stationary

    Returns:
    --------
        A calculation of the arrival rate for the specified time
        t. Units: (customers/hr)

    '''

    lmbda = lmbda_0*(1-A*np.cos(2*np.pi*t/T))

    return lmbda



        
class MonitoredResource(simpy.Resource):
    '''
    A class wrapper for the simpy.Resource object
    which obtains measuremnts for queue length
    and system populaion every change in instance
    
    Params:
    -------
    env: simpy.environment
        The simpy environment
    
    capacity: int
        The number of servers/capacity of the queue
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = []
        self.system_population = []

    #Updates resource measurments on every server request
    def request(self, *args, **kwargs):
        self.queue_length.append(len(self.queue))
        self.system_population.append(len(self.queue) + len(self.users))
        return super().request(*args, **kwargs)
    
    
##########
#Processes
##########

class queue_process:
    '''
    A template for an A/S/c (Arrival process/Service process/servers) format queueing process, 
    where distributions are specified by the arguments provided by a scenario type class.
    '''
    def __init__(self, identifier, env, args):
        '''
        Constructor method
        
        Params:
        -----
        identifier: int
            a numeric identifier for a customer.
            
        env: simpy.Environment
            the simulation environment
            
        args: Scenario
            Container class for the simulation parameters
            
        '''
        self.identifier = identifier
        self.env = env
        self.args = args
        
        # metrics
        
        self.arrival = -np.inf
        self.leave = -np.inf
        self.wait_service = -np.inf
        self.service_duration = -np.inf
        self.total_time = -np.inf
        
    def execute(self):
        '''
        Generates an instance of the A/S/c (Arrival process/Service process/servers) queueing process
        for an individual customer
        
        request server -> customer served -> customer leaves
    
        '''
        # record the time of arrival and queue length once entering the service queue
        
        self.arrival = self.env.now
        
        # request server
        with self.args.service.request() as req:
            yield req
            # record the waiting time for service
   
            self.wait_service = self.env.now - self.arrival     
            trace(f'Customer {self.identifier} is being served '
                  f'{self.env.now:.3f}',self.args.rout)
        
            # sample service duration.
            self.service_duration = self.args.service_dist.sample()
            yield self.env.timeout(self.service_duration)
            
            self.service_complete()
            
        # total time in system and queue length on leaving
        self.leave = self.env.now
        self.total_time = self.leave - self.arrival     
 
    def service_complete(self):
        '''
        Service complete event
        '''
        trace(f'Customer {self.identifier} service complete {self.leave:.3f}; '
              f'waiting time was {self.wait_service:.3f};'
              f'Service Duration was {self.service_duration}',self.args.rout)

        
#######
#Models
#######

class MMn_Model:
    '''
    Contains the model environment and data for a generic queue - MMn queue
    Simulation run condition is dictated by a specified results_collection_period

    '''
    def __init__(self, args):
        '''
        Constructor method
        
        Params:
        -----
        args: Scenario
            Container class for the simulation parameters
            
        '''
        #Initalise simpy environment and resources
        self.env = simpy.Environment()
        self.args = args
        self.init_resources()
        self.runtime = self.args.sim_period
        
        #Customer entity, arrival rate and times stores
        self.customers = []
        self.arrival_rates = []
        self.arrival_times = []

        self.rc_period = self.args.sim_period
        self.results = None
        
        #Initalise arrival rate function
        self.lambda_func = partial(lambda_func,
                                   T = self.args.sim_period, 
                                   lmbda_0 = self.args.arrival_rate_0, 
                                   A = self.args.process_coeff)

        
        

        
        
    def init_resources(self):
        '''
        Init the number of resources
        and records their use throughout the operation
        
        Resource list:
        1. Servers
         
        '''
        #Available servers
        self.args.service = MonitoredResource(self.env, 
                                              capacity=self.args.n_servers)
        
        
        
    def run(self):
        '''
        Conduct a single run of the model in its current 
        configuration
        
        
        Parameters:
        ----------
        results_collection_period, float, optional
            default = DEFAULT_RESULTS_COLLECTION_PERIOD
            
        Returns:
        --------
            None
        '''

        # setup the arrival generator process
        self.env.process(self.arrivals_generator())
        
        # run
        self.env.run(until=self.rc_period)   
        
        
    def arrivals_generator(self):  
        ''' 
        Simulate the arrival of customers to the model
        
        Non stationary arrivals implemented via Thinning acceptance-rejection 
        algorithm.
        '''

        for customer_count in itertools.count():

            # Calculates the arrival rate for the current simulation time
            t = self.env.now 
            lambda_t = self.lambda_func(t)
            trace(f'Current time {t} -- Arrival rate {lambda_t}',self.args.rout)
            
            #set to a large number so that at least 1 sample taken!
            u = np.inf
            
            interarrival_time = 0.0

            # reject samples if u >= lambda_t / lambda_max
            while u >= (lambda_t / self.args.lambda_max):
                interarrival_time += self.args.arrival_dist.sample()
                
                lambda_t = self.lambda_func(t + interarrival_time)

                u = self.args.thinning_rng.sample()
            
            
            self.arrival_rates.append(lambda_t)
            self.arrival_times.append(t)

            # iat
            yield self.env.timeout(interarrival_time)
            
            trace(f'customer {customer_count} arrives at: {self.env.now:.3f}',self.args.rout)
            
            new_customer = queue_process(customer_count, self.env, self.args) #Assigns new customer to queuing process instance
            self.customers.append(new_customer)

            # start the customer's queuing process for the customer
            self.env.process(new_customer.execute())
            

class GGn_Model:
    '''
    Contains the model environment and data for a special queue - G/S/c
    Here customers all arrive at once and are served sequentially. 
    The queue runs for as long as there are waiting customers,
    so there is no sim_period paramater.

    The a queue instance runtime is returned as a random variable R.

    '''
    def __init__(self, args):
        self.env = simpy.Environment()
        self.args = args
        self.init_resources()
        
        self.customers = []
        self.runtime = -np.inf
        self.results = None

        
        
    def init_resources(self):
        '''
        Init the number of resources
        and store in the arguments container object
        
        Resource list:
        1. Servers
         
        '''
        #Available servers
        self.args.service = MonitoredResource(self.env, 
                                              capacity=self.args.n_servers)
        
        
    def run(self,input):
        '''
        Conduct a single run of the model in its current 
        configuration
        
        
        Parameters:
        ----------
        results_collection_period, float, optional
            default = DEFAULT_RESULTS_COLLECTION_PERIOD
            
        Returns:
        --------
            None
        '''
        backlog = input

        if backlog > 0:

            self.env.process(self.backlog_service(backlog))
            # run
            self.env.run()
            #Obtain the final runtime of simulation

        self.runtime = self.env.now

        return self.runtime
        
        
    def backlog_service(self,B):  
        ''' 
        Simulates the arrival for the B customers.*

        *There may be a better way to do this, I just included a 0s
        pause between each arrival.  

        Paramaters
        ----------
        B: int
            The number of customers to be served. 

        '''

        #Generate new customers from previous backlog
        for customer_count in range(B):

            # Calculates the arrival rate for the current simulation time
            t = self.env.now 
           
            trace(f'Customer {customer_count} arrives at: {self.env.now:.3f}',self.args.rout)
            
            new_customer = queue_process(customer_count, self.env, self.args) #Assigns new customer to queuing process instance
            self.customers.append(new_customer)
            # start the customer's queuing process for the patient
            self.env.process(new_customer.execute())
            
            yield self.env.timeout(0) 
            

##########
#Model Templates
##########

class MMn_template:
    '''
    Container class for scenario parameters/arguments of the MMn queue
    
    M/M/n: Exponential Arrival and service times, n servers

    '''
    def __init__(self,arrival_rate_0=DEFAULT_ARRIVAL_RATE,
                 process_coeff=DEFAULT_PROC_COEF, 
                 n_servers=DEFAULT_SERVERS,
                 work_period = DEFAULT_WORK_PERIOD,
                 rout = False):
        
        '''
        Create a scenario to parameterise the simulation model
        
        Parameters:
        -----------
     
        arrival_rate_0: float
            Stationary arrival process rate

        process_coeff: float [0,1]
            Controls the non-stationarity of the arrival process
            0-1 stationary - non-stationary

        n_servers: int
            The number of servers

        sim_period: float (units: mins, default = 24*60 mins) 
            Length of the simulated queue service period

        '''

        #Marker
        self.mark = 'MMn'

        #Print Output?
        self.rout = rout

        #Model Type
        self.model = MMn_Model #Contains the model type, like DNA this thing contains itself and calls itself
        
        #sim run period

        self.sim_period = work_period

        #Arrival Process Params

        self.arrival_rate_0 = arrival_rate_0 #Lambda_0
        self.process_coeff = process_coeff  #A

        # count of each type of resource
        self.init_resourse_counts(n_servers)

        #Set by algorithm on initalisation of functions below
        # self.random_number_set = None
        # self.service_rate = None


    def init_resourse_counts(self, n_servers):
        '''
        Init the counts of resources to default values...
        '''
        self.n_servers = n_servers
        

    def init_sampling(self):
        '''
        Create the distributions used by the model and initialise 
        the random seeds of each.
        '''       
        # MODIFICATION. Better method for producing n non-overlapping streams
        seed_sequence = np.random.SeedSequence(self.random_number_set)
    
        # Generate n high quality child seeds
        self.seeds = seed_sequence.spawn(N_STREAMS)
        
        # create distributions
        
        self.service_mean = 60 / self.service_rate

        # Service Distb

        self.service_dist =  Exponential(self.service_mean, 
                                    random_seed=self.seeds[1])
        
        # init sampling for non-stationary poisson process
        self.init_nspp()
        
    def init_nspp(self):
        '''
        Initialises the arrival and thinning distbs for the non-stationary
        poisson process. Lambda max is double lambda_0 - hard coded
        
        '''
       
        # maximum arrival rate (smallest time between arrivals)
        self.lambda_max = lambda_func(t = self.sim_period/120,
                                      T = self.sim_period/60, 
                                      lmbda_0 = self.arrival_rate_0, 
                                      A = self.process_coeff)
        # thinning exponential
        self.arrival_dist = Exponential(60.0 / self.lambda_max,
                                        random_seed=self.seeds[8])
        
        # thinning uniform rng
        self.thinning_rng = Uniform(low=0.0, high=1.0, 
                                    random_seed=self.seeds[9]) 


    ## Model Setting Methods

    def set_service_rate_and_seed(self,service_rate, random_number_set= 12345):
        '''
        Allows for the update of the service rate when used in simulation
        optimisation. To maintain a 1d model service_var = service_mean/3
        '''
        self.random_number_set = random_number_set

        self.service_rate = service_rate
        
        self.init_sampling()


class GGn_template:
    '''
    Container class for scenario parameters/arguments
    
    G/G/n: General Arrival Times, Normal Distribution for Service Times
    '''
    def __init__(self, n_servers=DEFAULT_SERVERS, #Not needed in study but can be used later
                 service_min=DEFAULT_SERVICE_MIN,
                 rout = False):
        '''
        Create a scenario to parameterise the simulation model
        
        Parameters:
        -----------
        n_servers: int
            The number of servers
        
        service_min: float
            The truncation point for G (normal) service distribution. 
            This prevents negative service times

        rout: bool (Optional: default = False)
            Controls the simulation output messages
        '''
        #Marker
        self.mark = 'GGn'
        
        #Print Output?
        self.rout = rout


        #Model Type
        self.model = GGn_Model #Contains the model type, like DNA this thing contains itself and calls itself

        self.service_min = service_min

        # count of each type of resource
        self.init_resourse_counts(n_servers)


    def init_resourse_counts(self, n_servers):
        '''
        Init the counts of resources to default values...
        '''
        self.n_servers = n_servers
        

    def init_sampling(self):
        '''
        Create the distributions used by the model and initialise 
        the random seeds of each.
        '''       
        # MODIFICATION. Better method for producing n non-overlapping streams
        seed_sequence = np.random.SeedSequence(self.random_number_set)
    
        # Generate n high quality child seeds
        self.seeds = seed_sequence.spawn(N_STREAMS)
        
        # create distributions
        

        # Service Distb

        self.service_dist =  Normal(self.service_mean,
                                np.sqrt(self.service_var),
                                minimum=self.service_min,
                                random_seed=self.seeds[1])
        
   
    def set_service_rate_and_seed(self, service_rate, random_number_set=12345):
        '''
        Allows for the update of the service rate when used in simulation
        optimisation. To maintain a 1d model service_var = service_mean/3
        '''
        self.random_number_set = random_number_set

        self.service_rate = service_rate
        self.service_mean = 60/self.service_rate 
        
        self.service_var = self.service_mean/3 #I have removed the option to set the variance for now 
                                               #Thinking critically it is not requried until later
                                               #it should not be too difficult to implement then 
        self.init_sampling()


def model_constructor(model_type, model_args = DEFAULT_MODEL_PARAMS):

    '''
    Constructs model based upon specified type and the arguments for that type

    Params
    -----
    model_type: string (MMn/GGn)
        The queue type in Kendall notation

    model_args: dict (for form:see below)
        The model arguments for the corresponding queue.
        label names must be identical to the legend below.
        If not included default selected

        

    DEFAULT_MMn_ARGS = dict(arrival_rate_0 = DEFAULT_ARRIVAL_RATE,
                            process_coeff = DEFAULT_PROC_COEF,
                            work_period = DEFAULT_WORK_PERIOD)

    DEFAULT_GGn_ARGS = dict(service_min=DEFAULT_SERVICE_MIN)

    Returns
    -------

        template: MMn/GGn_template class instance
            The constructed template class.

      
    '''

    type_dial = dict(MMn = MMn_template,
                     GGn = GGn_template)


    template = type_dial[model_type]

    return template(**model_args)

                
        
################
#Data Collection
################

class SimulationMetric:
    '''
    End of run performance metric extraction class for a queueing model
    '''
    def __init__(self,metric, model, burn_in = 0):
        '''
        Constructor
        
        Params:
        ------
        model: queue_model_1
            The model.
        burn_in (optional): int
            Choose how much data the model discards
        '''
        self.model = model
        self.burn_in = burn_in
        self.args = model.args
        self.metric = metric

    
    def process_run_results(self):
        '''
        Calculates statistics at end of run.
        '''

        customers = self.model.customers[self.burn_in:]
        # Truncation of Measures fro m burn in

        metric_results = self.get_metric_results(self.metric, customers)
        
        return metric_results


    #You can modify this one at a later date, it can output the array as a tensor
    def get_metric_results(self, metric, customers):
        '''
        Returns the raw data for a specified metric from a single run of the simulation
        Params:
        -------
        metric: string
            list of all customer objects simulated.

        customers: customer object
            A customer instance (un)served during the simulation
        
        Returns:
        ------
        float
        '''
        d_out = 0

        if metric in ('wait_service','service_duration','total_time','arrival','leave'):
            
            t_data = np.array([getattr(c, metric) for c in customers if getattr(c, metric) > -np.inf])
            d_out = t_data

        elif metric in ('queue_length','system_population'):
            
            q_data = np.array(getattr(self.args.service, metric))
            d_out = q_data

        elif metric == 'backlog':
            q_data = np.array(getattr(self.args.service, 'queue_length'))
            d_out = q_data[-1]

        else:

            print(f'{metric} is not a valid performance metric')
        

        return d_out



##############
#Execute Simulation Meths
##############
    
def run_sim_pipeline(service_rates, seed, queue_1, queue_2):
    '''
    Performs a single run of the queue pipeline and returns the results.
    Queue 1 (M/M/1) is run for a period T_w. The backlog B after T_w, arrivals - served
    is recorded and fed into Queue 2 (G/G/1)
    
    Parameters:
    -----------
    service_rates: Matrix of floats (> 0)
        Service rates for this model instance. Included is the service rate
        for model 1 and 2

    scenario: Scenario object
        The scenario/paramaters used in the current experiment. Also includes the output indicator

    seed: int
        Seed used for the stochastic components of the queueing models i.e. arrival and service times
        
    queue_1: Template object, (default = DEFAULT_MODEL_1)
        The model class used for queue 1

    queue_1: Template object, (default = DEFAULT_MODEL_2)
        The model class used for queue 2
        
    Returns:
    --------
        runtime: Float
            The runtime of the second queue
    '''  
 
    
    # set random number set - this controls sampling for the run.
    queue_1.set_service_rate_and_seed(service_rates[0],random_number_set=seed)

    # create an instance of the model
    model = queue_1.model
    model_1 = model(queue_1)

    # run the model in node 1
    model_1.run()
   
    # run results to obtain backlog B
    B  = SimulationMetric('backlog',model_1).process_run_results()
    
    if queue_1.rout:
        print(f'Unserved: {B}') #Output unserved if rout is true


    #Returns zero runtime if queue 2 not triggered due to no backlog
    if B > 0:
        
        queue_2.set_service_rate_and_seed(service_rates[1],random_number_set = seed)

        # create an instance of the second model
        model = queue_2.model
        model_2 = model(queue_2)

        runtime =  model_2.run(B)

    else:
        runtime = 0

    if queue_2.rout:
        print(f'Runtime: {runtime}')
   
    return runtime




def run_custom_sim(service_rate, seed, queue, input = 0, metric = None):
    '''
    Performs a single run of the queue pipeline and returns the results.
    Queue 1 (M/M/1) is run for a period T_w. The backlog B after T_w, arrivals - served
    is recorded and fed into Queue 2 (G/G/1)
    
    Parameters:
    -----------
    service_rates: Matrix of floats (> 0)
        Service rates for this model instance. Included is the service rate
        for model 1 and 2

    scenario: Scenario object
        The scenario/paramaters used in the current experiment. Also includes the output indicator

    seed: int
        Seed used for the stochastic components of the queueing models i.e. arrival and service times
        
    queue_1: Template object, (default = DEFAULT_MODEL_1)
        The model class used for queue 1

    queue_1: Template object, (default = DEFAULT_MODEL_2)
        The model class used for queue 2
    input: int,
        If the model is a component further into the sim you can pass input to this function

    metric: str
        Name the metric that should be output from the queue
    #This has been included for a future implementation fo the code
        
    Returns:
    --------
        metric: Float
            The requested metric
    '''  
    
    #Procedures for using a type of model
    def run_MMn_type(model,input,metric):
        model.run()

        out = SimulationMetric('backlog',model).process_run_results()
        return out
        
    def run_GGn_type(model,input,metric):
        if input > 0:
            out = model.run(input)
        else:
            out = 0

        return out

    model_dial = dict(GGn = run_GGn_type,MMn = run_MMn_type)


    # set random number set - this controls sampling for the run.
    queue.set_service_rate_and_seed(service_rate,random_number_set=seed)
    model = queue.model
    model_1 = model(queue)

    run_model = model_dial[queue.mark]

    # create an instance of the model
    metric = run_model(model_1,input,metric)

   
    return metric


