
# SimQ.out 

### DEFAULT PARAMATERS

- RNG: DEFAULT_RNG_SET, N_STREAMS

- Model:
	>DEFAULT_ARRIVAL_RATE
	>DEFUALT_SERVICE_VAR
	>DEFAULT_SERVICE_MIN
	>DFAULT_PROC_COEFF


### DISTRIBUTIONS

- Exponential (class): for sampling interarrival times and service times in MMc queue
- Normal (class): for sampling service times in GGc queue
- Uniform (class): for thinning process in non-stationary arrival process for M(t)Mc

### Utils

- trace(func): utility function for printing output if TRACE=True

- lambda_func(func): A non-stationary (A>0)/stationary arrival(A=0) rate funciton

- MOnitoredResource (class-wrapper): Covers simpy's Resource class to obtain queu length measurements. 

### Processes

-queue_process (class): Creates a customer instance and commences their individual queueing process. 

### Model

- MMn_Model (class): Model instance for the MMc queue
- GGn_Model (class): Model instance for the GGc queue

### Model Templates:

- MMn_templace (class): Container class for queue template class and queue arguments for MMc queue.
-GGn_template (class): Container class for queue template class and queue arguments for GGc queue.

- model_constructor (func): Given a specified queue_type - a "MMn" or "GGn" string - and model arguments, this function creates an
instance of that class.

### Data Collection:

SimulationMetric (class): Handles the output from a queue simulation run(s)

### Execute Simulation Meths

Note: These are the functions used by SimQ_out.py to interact with the discrete event simulations

run_sim_pipeline (func): The method to run our toy problems complete, high-fidelity simulation - MM1 -> GG1, where both queues are modelled using discrete 
event simulations. 

run_custom_sim (func): The method lets us swap out one of the discrete event simulations in the toy problem- MM1 -> GG2 - for a low fidelity model. This 
is used during our multi-fidelity Bayesian Optimisation experiments. 
