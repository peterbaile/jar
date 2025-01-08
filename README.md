# 🫙 JAR: Join-Aware Multi-Table Retrieval

If you find our code, or the paper helpful, please cite the paper
```
@article{chen2024table,
  title={Is Table Retrieval a Solved Problem? Join-Aware Multi-Table Retrieval},
  author={Chen, Peter Baile and Zhang, Yi and Roth, Dan},
  journal={arXiv preprint arXiv:2404.09889},
  year={2024}
}
```

## Overview

* Query decomposition (`decomp.py`)
* Table-table compatibility scores (`compatibility.py`)
* Mixed-integer linear program (MIP) to retrieve the optimal set of tables (`ilp.py`)
* Baselines
  * DPR: Contriever (`contriever.py`)
  * DTR: TAPAS (`tapas.py`)

## Setup

To execute the MIP program, we recommend using [the Gurobi solver](https://docs.python-mip.com/en/latest/install.html#gurobi-installation-and-configuration-optional) for significantly faster execution.

Table-table compatibility scores generated from `compatibility.py` for Spider and Bird have been pre-computed and saved under `data/`.

The `dev_database` for Bird and Spider as well as the fine-tuned `TAPAS-large` checkpoints can be found in this [Google drive folder](https://drive.google.com/drive/folders/1PtLan7Guu98J42lqCxhZZc-EyZpvwWVk?usp=sharing).

You need to first execute `contriever.py` or `tapas.py` to obtain the coarse-grained relevance score and execute `contriever.py` to obtain fine-grained relevance score before you can run `ilp.py`.

Files with
```python
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--partition', type=int)
args = parser.parse_args()
```
support parallel execution. `num_partitions` specifies the the number of jobs executed in parallel. To run these files (e.g., `ilp.py`), you can use, for example,
```
python ilp.py -p 0 & python ilp.py -p 1 & ...
```

## Contact
If you have any questions or feedback, please send an email to peterbc@mit.edu.