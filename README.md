# Autoencoder experiments in Keras
The scripts are compatible with __Python 3.6__ and multiple GPUs execution.

## Usage
To run the program, execute the main script `nn_main.py`. The script supports the following command line arguments:
```
nn_main.py [-h] -c CONFIG [-l LOAD] [-T] [-t] [-e] [-s]
```
- `-c CONFIG`, `--config CONFIG` pass a configuration file specifying model architecture and training parameters.
- `-l LOAD`, `--load LOAD` pass a folder or a HDF5 file to load as weights or entire model.
- `-T`, `--train` execute training of the model.
- `-t`, `--test` execute testing routines.
- `-e`, `--err` redirect _stderr_ to log file.
- `-s`, `--save` archive configuration file (`-s`) and python scripts (`-ss`).
- `-h`, `--help` show the help message with description of the arguments.

As example, to execute a training of a new model, test the results and save the environment, run the following command from inside the `keras_ann/` folder:
```
$ python src/nn_main.py -c config/cnfg_file -Ttss
```