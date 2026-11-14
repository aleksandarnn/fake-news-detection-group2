"""Fake news detection on the LIAR dataset — Group 2, University of Twente.

Modules
-------
data        Loading the LIAR splits, label mapping, text transforms.
baselines   TF-IDF + linear model experiments and the two diagnostics.
distilbert  DistilBERT fine-tuning.
figures     Regenerates every figure in `figures/` from tracked artifacts.
"""

__all__ = ["data", "baselines", "distilbert", "figures"]
