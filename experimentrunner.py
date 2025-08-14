import argparse
from execute import ExperimentConfig, OptimizerRunner


parser = argparse.ArgumentParser()
    
parser.add_argument("--objective_function", type=str)
parser.add_argument("--objective_dim", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--n_init", type=int) 
parser.add_argument("--n_starts", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--optimizer", type=str) 
parser.add_argument("--p", type=float, default=None)
parser.add_argument("--gradient_learning_samples", type=int) 
parser.add_argument("--step_size", type=float, default=0.01)
    
args = parser.parse_args()
    
config = OptimizerRunner(
    objective_function=args.objective_function,
    objective_dim=args.objective_dim,
    iterations=args.iterations,
    n_init=args.n_init,
    n_starts=args.n_starts,
    seed=args.seed,
    pick_optimizer=args.optimizer,
    p=args.p,
    gradient_learning_samples=args.gradient_learning_samples,
    step_size = args.step_size
    )
    
    
    
experiment = ExperimentConfig(config)
experiment.experiment_setup()
    


#python experimentrunner.py --objective_function Branin --objective_dim 2 --iterations 20 --n_init 10 --n_starts 3 --seed 10 --optimizer MPDwithCOCOB --p 0.85 --gradient_learning_samples 1 --step_size 0.01

