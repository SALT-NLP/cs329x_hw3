# HW3: Building and Evaluating Human-AI Interaction

**Name:** [Your Name]

**SUNet ID:** [Your ID]


## CoGym (25 points)

### Question 1.1 (15 points)

Produce 3 travel plans with cogym!  Please be creative and create travel plans that are actually interesting to you or have interesting constraints (budget, activities, multi-city, etc.)

#### Travel Plan One

```
[PASTE YOUR FIRST COGYM TRAVEL PLAN HERE]
```

#### Travel Plan Two

```
[PASTE YOUR SECOND COGYM TRAVEL PLAN HERE]
```

#### Travel Plan Three

```
[PASTE YOUR THIRD COGYM TRAVEL PLAN HERE]
```

### Question 1.2 (5 points)

Evaluate these travel plans.  Which one is the best?  Which was the worst?  Score each on quality from (1-5) using your own criteria. (We are looking for a score of 1-5 for each travel plan).

**Travel Plan One Score:** X/5

**Travel Plan Two Score:** X/5

**Travel Plan Three Score:** X/5

### Question 1.3 (5 points)

Explain what criteria you were using when doing this evaluation.  What mattered?  (2-3 sentences).

```
[WRITE YOUR 2-3 sentence explanation of what evaluation criteria mattered to you here.]
```

## Building Automatic Evaluators to Approximate Human Judgement (50 points + 10 extra credit)

### Question 2.1 (10 points)

Copy and paste your prompt and Pearson correlation here.

```
[PASTE YOUR PROMPT HERE]
```

Pearson correlation: X.XXX ± X.XXX

### Question 2.2 (10 points)

Paste your implmentation of `generate_criteria` here.

```python
# Generate criteria from non-descriptive human feedback
# Inputs:
#   train_df: a dataframe of training examples with input, output, and score columns
#   human_criteria: a string of the human criteria for the task
#   seed: an integer for the random seed
# Outputs:
#   A list of strings of the criteria.
def generate_criteria(train_df, human_criteria="Unknown", seed=42) -> list[str]:
    # TODO: PASTE YOUR IMPLEMENTATION HERE
```

Paste the criteria you generated for each dataset here.

```
[CoGym Evaluation Criteria]
```

```
[HelpSteer2 Evaluation Criteria]
```

```
[SimpEval Evaluation Criteria]
```

### Question 2.3 (10 points)

Paste your implementation of `regress_criteria` here.

```python
# Run the LLM judges on the training dataset and return the regression coefficients and intercept
# Inputs:
#   train_df: a dataframe of training examples with input, output, and score columns
#   criteria_list: a list of strings of the LLM as a Judge criteria to regress on
# Outputs:
#   A tuple of the regression coefficients (list[float]) and intercept (float)
def regress_criteria(train_df, criteria_list) -> Tuple[List[float], float]:
    # TODO: PASTE YOUR IMPLEMENTATION HERE
```

And your Pearson Correlation on CoGym

Pearson correlation: X.XXX ± X.XXX

### Question 2.4 (20 points + 10 extra credit for top 3 submissions)

Paste your implementation of `generate_evaluator` here.

```python
# Given a train_df and human_criteria, generate an automatic evaluator that can be used to score a model's output
# Inputs:
#   train_df: a dataframe of training examples with input, output, and score columns
#   human_criteria: a string of the human criteria for the task (e.g. "travel plan quality", "helpfulness", "simplification quality")
# Outputs:
#   A callable function that takes an input and output and returns a score
def generate_evaluator(train_df, human_criteria="Unknown") -> Callable[[str, str], float]:
    # TODO: Replace this implementation with your code here!  This implements the generation + regression method from above
```

Paste your results on CoGym, Helpsteer2, and SimpEval below.

```
[Cogym] Pearson correlation:  X.XXX ± X.XXX
[HelpSteer2] Pearson correlation:  X.XXX ± X.XXX
[SimpEval] Pearson correlation:  X.XXX ± X.XXX
Average Pearson correlation:  X.XXX
```

Finally please write up a 1-2 paragraph explanation of your approach.  Be sure to cite any papers that introduce similar methods if you took inspiration from the literature.

```
[WRITE YOUR 1-2 PARAGRAPH EXPLANATION OF YOUR APPROACH HERE]
```