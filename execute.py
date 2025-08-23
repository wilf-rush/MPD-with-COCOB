import os
import warnings
import torch
from botorch.test_functions import Branin, Hartmann, Ackley, Levy, Griewank, Rosenbrock, Rastrigin
import torch
import numpy as np
from botorch.exceptions import BadInitialCandidatesWarning
from main import GenerateData, MPD, MPDwithCOCOB, ExecuteOptimizer
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
SMOKE_TEST = os.environ.get("SMOKE_TEST")


class OptimizerRunner:
    def __init__(
            #default values for the inputs
            self,
            objective_function: str= "Branin",
            objective_dim: int = 2,
            iterations: int = 100,
            n_init: int = 10,
            n_starts: int = 1,
            pick_optimizer: str = "MPD",
            p: float = 0.5,
            gradient_learning_samples: int = 1,
            alpha: int = 100,
            reset_num: int = 1,
            if_norm: str = "no",
            step_size: float = 0.01,
            ):
        
        self.objective_function = objective_function
        self.objective_dim = objective_dim
        self.iterations = iterations
        self.n_init = n_init
        self.n_starts = n_starts
        self.pick_optimizer = pick_optimizer
        self.p = p
        self.gradient_learning_samples = gradient_learning_samples
        self.alpha = alpha
        self.reset_num = reset_num
        self.if_norm = if_norm
        self.step_size = step_size


#configure the experiments
class ExperimentConfig:
    def __init__(self,config :OptimizerRunner):

        self.config = config
        self.generate_and_train = GenerateData(device=device, dtype=dtype)
        self.objective = self.objective_picker()

        ##### storage
        self.paths = []
        self.moves = []
        self.rates = []
        self.times = []
        #####


    def objective_picker(self):
        if self.config.objective_function == "Branin": 
             return Branin(negate=True).to(dtype=dtype,device=device)
        if self.config.objective_function == "Hartmann": 
             return Hartmann(negate=True).to(dtype=dtype,device=device)
        if self.config.objective_function == "Ackley": 
             return Ackley(negate=True).to(dtype=dtype,device=device)
        if self.config.objective_function == "Levy": 
             return Levy(negate=True).to(dtype=dtype,device=device)
        if self.config.objective_function == "Griewank": 
             return Griewank(negate=True).to(dtype=dtype,device=device)
        if self.config.objective_function == "Rosenbrock": 
             return Rosenbrock(negate=True).to(dtype=dtype,device=device)
        if self.config.objective_function == "Rastrigin": 
             return Rastrigin(negate=True).to(dtype=dtype,device=device)
        
    

    #setup the experiment
    def experiment_setup(self):
        train_xs, train_ys = self.generate_and_train.generate_shared_initial_data(
            self.objective,
            self.config.n_init,
        )

        starting_points = self.generate_and_train.generate_sobol_starting_points(
          self.objective,
          self.config.n_starts,
        )


        ###could add another loop to cycle through step size etc
        for i in range(self.config.n_starts):
            self.run_experiment(train_xs, train_ys, starting_points[i])
        
        
        self.print_path()
        
      


    def run_experiment(self,train_xs,train_ys,starting_point):
        gp_model, gp_trainer = self.generate_and_train.get_data_and_gp(
            self.objective, train_xs, train_ys
        )

        if self.config.pick_optimizer == "MPD":
            optimizer = MPD(
                objective=self.objective,
                gp_model=gp_model,
                gp_trainer=gp_trainer,
                function_bounds=self.objective.bounds,
                min_descent_prob=self.config.p,
                gradient_learning_samples=self.config.gradient_learning_samples,
                move_counter=0,
                iter_counter=0,
                step_size=self.config.step_size
            )
        if self.config.pick_optimizer == "MPDwithCOCOB":
            optimizer = MPDwithCOCOB(
                objective=self.objective,
                gp_model=gp_model,
                gp_trainer=gp_trainer,
                function_bounds=self.objective.bounds,
                min_descent_prob=self.config.p,
                gradient_learning_samples=self.config.gradient_learning_samples,
                move_counter=0,
                iter_counter=0,
                x_1=starting_point,
                alpha = self.config.alpha,
                if_norm = self.config.if_norm,
                reset_num = self.config.reset_num
            )


        execute = ExecuteOptimizer(
            optimizer=optimizer,
            initial_x= starting_point, 
            iterations=self.config.iterations
        )

        opt = execute.run_optimizer()


        ####### storage (this is what was missing)
        self.paths.append(np.array(execute.iteration_values).flatten())
        self.moves.append(np.array(execute.move_lengths).flatten())
        self.rates.append(np.array(execute.effective_learning_rates).flatten())
        self.times.append(execute.time)
        ########

        return opt
    


    def print_path(self):
        np.set_printoptions(precision=3, suppress=True)

        for i, (path, move, rate, t) in enumerate(zip(self.paths, self.moves, self.rates, self.times)):
            print(f"\n--- Seed {i+1} ---")
            print("Path:", path)
            print("Moves:", move)
            print("Rates:", rate)
            print("Time:", t)

      


























