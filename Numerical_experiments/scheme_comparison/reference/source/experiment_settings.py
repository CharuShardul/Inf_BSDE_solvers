"""Training lengths shared by the solvers and current experiment drivers.

See README.md for the distinct settings of the historical article figures.
"""

NN_PICARD_EPOCHS = 200  # Epochs per Picard iteration, not across all iterations.
DIRECT_N_STEPS = 3000  # Optimizer updates; the direct solver does not use epochs.
