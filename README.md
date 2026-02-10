# Scaling with Multiple Scaling Factors in Time Series Searching

<img src="figures/psdtw-intuition.pptx.svg" alt="Intuition of PSD." width="400">

- It shows the intuition of the necessity of our novel distance measure PSD (Piecewise Scaling Distance).

## Installation

```
conda create -n ksfdtw python=3.12
conda activate ksfdtw
conda install -c conda-forge dtaidistance
pip install pandas
conda install -c conda-forge tslearn
pip install tqdm # For progress bar
pip install pyts
conda install -c conda-forge sktime  
conda install -c conda-forge aeon
conda install h5py
# Export your active environment to a new file:
conda env export > environment.yml
# Deactivate the environment
conda deactivate
# Delete the environment if needed.
conda env remove -n ksfdtw

# Or readers can make a new environment from our environment. 
# But different platforms may have their own platform-specific packages that may cause error if importing `environments.yml` directly.
conda env create --name envname --file=environments.yml
```

## Project Structure

- Data Preparation: [data_processing.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/02_searching.ipynb)
- Query Experiment: [querying.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/02_searching.ipynb)

  <!-- ```bash
  .
  ├── code
  │   ├── class_representative_fig.ipynb
  │   ├── data_processing.ipynb
  │   ├── dtw_bands_fig.ipynb
  │   ├── ksfdtw
  │   │   ├── distance_measures.py
  │   │   ├── lower_bounds.py
  │   │   └── utils.py
  │   ├── lb_keogh_vs_lb_shen_fig.ipynb
  │   ├── main.ipynb
  │   ├── querying_ex.ipynb
  │   ├── querying.ipynb
  ├── data
  ├── data_intermediate
  ├── environment.yaml
  ├── figures
  ├── README.md
  └── results
  ``` -->

<!-- ## Figures/Tables in the Paper
- Figures 1, 4: [main.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/main.ipynb)
- Figure 3: [dtw_bands_fig.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/dtw_bands_fig.ipynb)
- Figure 5: [lb_keogh_vs_lb_shen_fig.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/lb_keogh_vs_lb_shen_fig.ipynb)
- Figure 6: [class_representative_fig.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/class_representative_fig.ipynb)
- Figure 7: [data_processing.ipynb](https://github.com/cyuab/k-scaling-factor-dtw/blob/main/code/data_processing.ipynb) -->

## Run the scripts in tmux to allow multiple runnings
Create a tmux session
```
# List tmux sessions
$ tmux ls
# Create new session
# tmux new -s <session-name>
$ tmux new -s 02_searching1
# Switch to bash in tmux
$ bash
$ conda activate ksfdtw
$ cd code
$ python 02_searching1.py
# Detach from tmux session
# https://superuser.com/questions/249659/how-to-detach-a-tmux-session-that-itself-already-in-a-tmux
Press Ctrl+B then D
```
Manage tmux sessions
```
# Attach back to a specific tmux session
$ tmux attach -t <session-name>
# Kill a tmux session
$ tmux kill-session -t <session-name>
# Kill all tmux sessions
# https://askubuntu.com/questions/868186/how-to-kill-all-tmux-sessions-or-at-least-multiple-sessions-from-the-cli
$ tmux kill-server
```
### References

- [A beginner's guide to tmux](https://www.redhat.com/en/blog/introduction-tmux-linux)
- [Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html) (In Chinese)


## Resources

1. [aeon](https://www.aeon-toolkit.org/en/stable/index.html)
    - [Distances - aeon 1.2.0 documentation](https://www.aeon-toolkit.org/en/stable/api_reference/distances.html)
    - [dtw_distance - aeon 1.2.0 documentation](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.distances.dtw_distance.html)
1. [DTAIDistance](https://dtaidistance.readthedocs.io/en/latest/index.html)
    - [Dynamic Time Warping (DTW) — DTAIDistance 2.3.9 documentation](https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html)
1. [tslearn](https://tslearn.readthedocs.io/en/stable/index.html)
    - [tslearn.metrics — tslearn 0.6.4 documentation](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.metrics.html#module-tslearn.metrics)
    - [Dynamic Time Warping — tslearn 0.6.4 documentation](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html)
1. [pyts](https://pyts.readthedocs.io/en/stable/index.html)
    - [pyts.metrics.dtw — pyts 0.13.0 documentation](https://pyts.readthedocs.io/en/stable/generated/pyts.metrics.dtw.html)
1. [sktime](https://www.sktime.net/en/stable/)
    - [dtw_distance — sktime documentation](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.distances.dtw_distance.html)
1. FastDTW
    - [Original Java implementation](https://github.com/rmaestre/FastDTW)
    - [Python implementation](https://github.com/slaypni/fastdtw)
1. [dtw-python · PyPI](https://pypi.org/project/dtw-python/)
1. [dtw · PyPI](https://pypi.org/project/dtw/)