import os
import warnings
import time
import torch
from torch.distributions import Normal
import gpytorch
import gpytorch
import torch
from torch.quasirandom import SobolEngine
from torch.distributions import Normal
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.transforms import unnormalize
from gpytorch.utils.cholesky import psd_safe_cholesky
from abc import ABC, abstractmethod

from acquisitionfunction import DownhillQuadratic, optimize_acqf_custom_bo
from model import DerivativeExactGPSEModel
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
SMOKE_TEST = os.environ.get("SMOKE_TEST")



class BaseOptimizer(ABC):
    def __init__(self, objective, gp_model, gp_trainer, function_bounds, min_descent_prob, gradient_learning_samples, move_counter, iter_counter=0):
        
        self.objective = objective
        self.gp_model = gp_model
        self.gp_trainer = gp_trainer
        self.function_bounds = function_bounds
        self.min_descent_prob = min_descent_prob
        self.gradient_learning_samples = gradient_learning_samples
        self.move_counter = move_counter
        self.iter_counter = iter_counter
        self.delta = self.calculate_delta(function_bounds)

    def calculate_delta(self, function_bounds):
        width = function_bounds[1] - function_bounds[0]
        return 0.2 * width
    
    def learn_gradients(self, x):
        acq_fun = DownhillQuadratic(self.gp_model)
        lower_local = torch.max(x.squeeze() - self.delta, self.function_bounds[0])
        upper_local = torch.min(x.squeeze() + self.delta, self.function_bounds[1])
        local_bounds = torch.stack([lower_local, upper_local])
        for m in range(self.gradient_learning_samples):
            acq_fun.update_theta_i(x) 
            z, acq_value = optimize_acqf_custom_bo(acq_func=acq_fun, bounds=local_bounds,q=1,num_restarts=2,raw_samples=20)
            y_z = self.objective(z)
            self.gp_model.append_train_data(z, y_z)
        self.gp_trainer.train(self.iter_counter)

    @abstractmethod
    def update_gp(self, x, y):
        pass

    @abstractmethod  
    def update_step(self, x):
        pass










class GenerateData:
    def __init__(self, device=None, dtype=None):
        self.device = device 
        self.dtype = dtype

    def generate_shared_initial_data(self, objective, n_init):
        sobol = SobolEngine(dimension=objective.dim, scramble=True, seed=10)
        xs = sobol.draw(n=n_init).to(dtype=self.dtype, device=self.device)
        train_xs = unnormalize(xs, objective.bounds)
        train_ys = objective(train_xs)
        return train_xs, train_ys

    def get_data_and_gp(self, objective, train_xs, train_ys):
        gp_model = DerivativeExactGPSEModel(objective.dim)
        for i in range(len(train_xs)):
            gp_model.append_train_data(train_xs[i:i+1], train_ys[i:i+1])
        gp_trainer = GPTrain(gp_model)
        gp_trainer.train(0)
        return gp_model, gp_trainer
    
    def generate_sobol_starting_points(self, objective, n_starts):
        sobol = SobolEngine(dimension=objective.dim, scramble=True, seed=10)
        sobol_points = sobol.draw(n=n_starts).to(dtype=self.dtype, device=self.device)
        starting_points = unnormalize(sobol_points, objective.bounds)
        return starting_points
    









class GPTrain:
    def __init__(self, model, first_run: int = 500, training_its: int = 50, lr: float = 0.1):
        self.model = model
        self.first_run = first_run
        self.training_its = training_its
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    def train(self, tc: int):
        self.model.train()
        self.model.likelihood.train()

        if tc == 0:
            training_its = self.first_run
        else:
            training_its = self.training_its

        for _ in range(training_its):
            self.optimizer.zero_grad()
            output = self.model(self.model.train_inputs[0])
            loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            self.optimizer.step()










class MPD(BaseOptimizer):
    def __init__(self, objective, gp_model, gp_trainer, function_bounds, min_descent_prob, gradient_learning_samples, move_counter, iter_counter,step_size):

        super().__init__(objective, gp_model, gp_trainer, function_bounds, min_descent_prob, gradient_learning_samples, move_counter,iter_counter)
        
        self.step_size = step_size
        self.xs = []
        self.ys = []
    
    def update_gp(self, x, y):
        self.gp_model.append_train_data(x, y)
        self.gp_trainer.train(self.iter_counter)

    def calculate_direction_and_prob(self, x):
        self.gp_model.eval()
        self.gp_model.likelihood.eval()
        with torch.no_grad():
            mean_d, var_d = self.gp_model.posterior_derivative(x)
            mean_d = mean_d.squeeze(0)
            var_d = var_d.squeeze(0)
            chol_d = psd_safe_cholesky(var_d)  
            y = torch.linalg.solve_triangular(chol_d, mean_d.unsqueeze(-1), upper=False)
            v_star = torch.linalg.solve_triangular(chol_d.transpose(-2, -1), y, upper=True).squeeze(-1)
            mpd = Normal(0, 1).cdf(torch.matmul(v_star, mean_d).sqrt()).item()
        return v_star, mpd

    def update_step(self, x):
        v_star , mpd = self.calculate_direction_and_prob(x)
        self.move_counter = 0 
        while mpd > self.min_descent_prob and self.move_counter < 10_000: 
            self.move_counter += 1
            x_new = x + self.step_size * v_star
            x = x_new.detach()
            within_bounds = torch.all(x >= self.function_bounds[0]) and torch.all(x <= self.function_bounds[1])
            if within_bounds:
                self.xs.append(x.clone())
                self.ys.append(self.objective(x))
            v_star , mpd = self.calculate_direction_and_prob(x)
        return x
    









class MPDwithCOCOB(BaseOptimizer):
    def __init__(self, objective, gp_model, gp_trainer, function_bounds, min_descent_prob, gradient_learning_samples,move_counter, iter_counter, x_1, a=100):
        
        super().__init__(objective, gp_model, gp_trainer,function_bounds, min_descent_prob, gradient_learning_samples,move_counter,iter_counter)
        
        self.a = a
        self.x_1 = x_1
        dim = objective.dim

        self.xs = []
        self.ys = []
        
        self.L = torch.zeros(1, dim, dtype=dtype, device=device)
        self.G = torch.zeros(1, dim, dtype=dtype, device=device) 
        self.reward = torch.zeros(1, dim, dtype=dtype, device=device)
        self.theta = torch.zeros(1, dim, dtype=dtype, device=device)
        
    def update_gp(self, x, y):
        self.gp_model.append_train_data(x, y)
        self.gp_trainer.train(self.iter_counter)

    def reset_coin_betting_variables(self):
        dim = self.objective.dim
        self.L = torch.zeros(1, dim, dtype=dtype, device=device)
        self.G = torch.zeros(1, dim, dtype=dtype, device=device) 
        self.reward = torch.zeros(1, dim, dtype=dtype, device=device)
        self.theta = torch.zeros(1, dim, dtype=dtype, device=device)
        
    def calculate_direction_and_prob_COCOB(self, x):
        self.gp_model.eval()
        self.gp_model.likelihood.eval()
        with torch.no_grad():
            mean_d, var_d = self.gp_model.posterior_derivative(x)
            mean_d = mean_d.squeeze(0)
            var_d = var_d.squeeze(0)
            chol_d = psd_safe_cholesky(var_d)  
            y = torch.linalg.solve_triangular(chol_d, mean_d.unsqueeze(-1), upper=False)
            v_star = torch.linalg.solve_triangular(chol_d.transpose(-2, -1), y, upper=True).squeeze(-1)
            mpd = Normal(0, 1).cdf(torch.matmul(v_star, mean_d).sqrt()).item()
        return v_star, mpd

    def update_step(self, x):
        #calculate mpd
        v_star , mpd = self.calculate_direction_and_prob_COCOB(x)
        self.move_counter = 0
        while mpd > self.min_descent_prob and self.move_counter < 10_000:
            self.move_counter += 1
            
            #coin betting move
            grad = v_star
            #update maximum observed scale
            self.L = torch.max(self.L, torch.abs(grad)).detach()
            #update sum of absolute values of gradients
            self.G += torch.abs(grad).detach()
            #update reward
            self.reward = torch.max(self.reward + (x - self.x_1) * grad, torch.zeros_like(self.reward)).detach()
            #update sum of the gradients
            self.theta += grad.detach()
            #calculate xs values - betted amounts
            x_new = self.x_1 + (self.theta / (self.L * torch.max(self.G + self.L, self.a * self.L))) * (self.L + self.reward)
            x = x_new.detach()
            
            within_bounds = torch.all(x >= self.function_bounds[0]) and torch.all(x <= self.function_bounds[1])
            if within_bounds:
                self.xs.append(x.clone())
                self.ys.append(self.objective(x))
            
            v_star , mpd = self.calculate_direction_and_prob_COCOB(x)

        return x
    









class ExecuteOptimizer:
    def __init__(self, optimizer, initial_x, iterations,iteration_values=[],move_lengths=[],time=0):
        self.optimizer = optimizer
        self.initial_x = initial_x
        self.iterations = iterations    
        self.iteration_values = iteration_values
        self.move_lengths = move_lengths
        self.time = time

#### removes the excess points and only keep the best x and y
    def remove_excess_points(self):
        ys = torch.cat([y.reshape(1) for y in self.optimizer.ys])
        best_point = ys.argmax()
        best_x = self.optimizer.xs[best_point].clone()
        best_y = self.optimizer.ys[best_point].clone()
        self.optimizer.xs = [best_x]
        self.optimizer.ys = [best_y]



    def run_optimizer(self):
        self.iteration_values = []
        self.move_lengths = []
        self.time = 0

        #get the initial x
        x = self.initial_x.clone()
        print(f'Starting location: {x}')
    

        self.optimizer.xs.append(x.clone())
        self.optimizer.ys.append(self.optimizer.objective(x).unsqueeze(0))

        start = time.time()




        #optimization loop
        for i in range(self.iterations):

            self.optimizer.iter_counter += 1
            print(self.optimizer.iter_counter)
            
            #reset cocob variables to 0 
            if isinstance(self.optimizer, MPDwithCOCOB):
                self.optimizer.reset_coin_betting_variables()

            #use best x so far
            ys = torch.cat([y.reshape(1) for y in self.optimizer.ys])
            best_point = ys.argmax() 
            x = self.optimizer.xs[best_point].clone()
            
            if x.dim() == 1:
                x = x.unsqueeze(0)

            #evaluate objective          
            y = self.optimizer.objective(x)
            self.iteration_values.append(y.item())
            print(y)
            
            
            #update GP with new observation
            self.optimizer.update_gp(x, y)

            #acquisition function
            self.optimizer.learn_gradients(x)

            #update step
            x = self.optimizer.update_step(x)
            self.move_lengths.append(self.optimizer.move_counter)

            #make sure best x is returned
            self.remove_excess_points()
            best_x = self.optimizer.xs[0]

        end = time.time()
        self.time = end - start
            
        return best_x
       
    
    