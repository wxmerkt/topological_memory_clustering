# Quadrotor Example

## Dataset generation

In order to generate the dataset for the cross environment, make sure you have Exotica installed:
```bash
sudo apt install -y ros-$ROS_DISTRO-exotica-examples
```

To run the config for a single optimisation (e.g., to tune the optimal control settings), use:
```
python$ROS_PYTHON_VERSION quadrotor_single_solve.py
```

You can change the environment config and other settings in the sampling script (`quadrotor_parallel_sampling.py`).
Then run generate the dataset in parallel (uses all available cores):

```
python$ROS_PYTHON_VERSION quadrotor_parallel_sampling.py
```

This generates a `parallel_samples.npz` file with the dataset. You can view the dataset by running:
```
python$ROS_PYTHON_VERSION visualise_dataset.py parallel_samples.npz
```

## Memory clustering

Start a Jupyter session:
```
jupyter lab .
```

and open the following notebooks:

- [./\[Master-random-starts\]\ Analyse\ quadrotor\ samples.ipynb](Master-random-starts - Analyse quadrotor samples)

## Notes

- The datasets in the paper were generated using the explicit Euler integrator (`Integrator="RK1"` in the Exotica configs). Omitting this specification uses a symplectic integrator which is more stable. This feature was not fully available in Exotica when the paper was first written.
