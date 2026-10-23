"""Existing project entry point for the paper-style three-module baseline.

Run with `python fl_basic.py`; parameters are controlled with KD_* environment
variables documented in SLURM_GUIDE.md.
"""
from organizer import Organizer

if __name__ == "__main__":
    org = Organizer()
    org.federated_training_with_distillation()
    #org.federated_training_basic()
