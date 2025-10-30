from typing import Callable
from helpers import *

# Given a train_df and human_criteria, generate an automatic evaluator that can be used to score a model's output
# Inputs:
#   train_df: a dataframe of training examples with input, output, and score columns
#   human_criteria: a string of the human criteria for the task (e.g. "travel plan quality", "helpfulness", "simplification quality")
# Outputs:
#   A callable function that takes an input and output and returns a score
def generate_evaluator(
    train_df, human_criteria="Unknown"
) -> Callable[[str, str], float]:
    # TODO: Replace this implementation with your code here!  This implements the generation + regression method from above
    def evaluate(input: str, output: str) -> float:
        return 0.0
    return evaluate