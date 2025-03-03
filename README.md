# Cellular Automata Analysis

## Install dependencies

Ensure you have Python installed. Then, install the required Python packages using the following command:

```sh
pip install -r requirements.txt
```

## Setup

Run the `setup.py` script to create the necessary directories for storing data:

```sh
python setup.py
```

## Generate Trajectories

To generate trajectories for Elementary Cellular Automata (ECA) rules, run the `eca_generation.py` script. This will create a pickle file containing the trajectories in the specified directory.

```sh
python eca_generation.py
```

### Using the Functions

If you want to use the functions directly in your own scripts, you can import them from `eca_generation.py`. Here is a brief description of the main functions with example usage:

#### `CA_run(initial_state, n_steps, rule_number)`

Simulates a one-dimensional cellular automaton (CA) for a given number of steps.

**Arguments:**
- `initial_state` (list or array-like): The initial state of the CA, a list or array of integers (0 or 1).
- `n_steps` (int): The number of steps to simulate.
- `rule_number` (int): The rule number (0-255) that defines the CA's behavior, based on Wolfram's rule numbering.

**Returns:**
- `numpy.ndarray`: A 2D array where each row represents the state of the CA at a given step.

**Example:**
```python
from eca_generation import CA_run
initial_state = [0, 1, 0, 1, 0]
n_steps = 10
rule_number = 30
result = CA_run(initial_state, n_steps, rule_number)
print(result)
```

#### `eca_trajectories(_width, _rules, _ICs, _path=".", _out_file_name="trajectories", _num_workers=4)`

Generate and save trajectories of 1D ECA based on the rules' number and a set of initial conditions (ICs).

**Arguments:**
- `_width` (int): The width of the cellular automaton.
- `_rules` (list): A list of ECA rule numbers to generate trajectories for.
- `_ICs` (list or array-like): A collection of lists or array-like data structures containing the array-like of binaries (0s or 1s) defining initial conditions of the ECA.
- `_path` (str, optional): The directory path where the output file will be saved. Default is the current directory.
- `_out_file_name` (str, optional): The base name of the output file. Default is "trajectories".
- `_num_workers` (int, optional): The number of worker processes to use for parallel processing. Default is 4.

**Returns:**
- `None`

**Example:**
```python
from eca_generation import eca_trajectories
import numpy as np
from itertools import product

width = 5
rules = [30, 45, 73]
ICs = [np.array(bits) for bits in product([0, 1], repeat=width)]
eca_trajectories(width, rules, ICs, _path=".", _out_file_name="trajectories", _num_workers=2)
```
