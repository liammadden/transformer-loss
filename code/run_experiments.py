import os
import torch
from experiment import Experiment
from run import Run

# Get Path
path = os.path.dirname(os.getcwd())

# Create folders for experiments and plots
if not os.path.exists(os.path.join(path, "experiments")):
    os.makedirs(os.path.join(path, "experiments"))
if not os.path.exists(os.path.join(path, "plots")):
    os.makedirs(os.path.join(path, "plots"))

# Set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

### Set parameters ###
sequence_length = 100
dataset = "tinystories"
n = 100
epochs = 1000
batch_size = "full"
plot_only = False  # change to True if you want to plot existing experimental results, assuming experiment pkl file already exists
m_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90]

runs = []
# Create runs
for m in m_vals:
    run = Run(m=m, d=m)
    runs.append(run)

# Run experiment
ex = Experiment(
    n = n,
    sequence_length=sequence_length,
    dataset=dataset,
    batch_size=batch_size,
    epochs=epochs,
    runs=runs,
)
ex.run_experiment(plot_only=plot_only, path=path, device=device)
