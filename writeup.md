# HW3: Building and Evaluating Human-AI Interaction

**Name:** [Your Name]

**SUNet ID:** [Your ID]


## Collecting Human-Agent Collaboration Traces (X points) <!-- TODO(staff): set points -->

### Question 1.1 (X points)

Write down the two planning queries you worked on with Coco, the session ids you exported to `traces/coco_traces.json`, and a short description of how each collaboration went (2-3 sentences each).

#### Session One

**Query:** [YOUR FIRST QUERY]

**Session id:** [SESSION ID]

```
[HOW DID THE COLLABORATION GO?]
```

#### Session Two

**Query:** [YOUR SECOND QUERY]

**Session id:** [SESSION ID]

```
[HOW DID THE COLLABORATION GO?]
```

### Question 1.2 (X points)

#### First Persona

```
[PASTE YOUR FIRST PERSONA HERE]
```

#### Final Persona

```
[PASTE YOUR FINAL PERSONA HERE]
```

#### Comparison

For three turns of your choice, paste Coco's previous message (you can truncate it), your real message, and Persimmon's prediction under your final persona.

```
[TURN 1: Coco / You (real) / Persimmon]
```

```
[TURN 2: Coco / You (real) / Persimmon]
```

```
[TURN 3: Coco / You (real) / Persimmon]
```

#### Reflection

What did you change between persona versions and why? In what ways does the simulated user still differ from you? (4-6 sentences)

```
[WRITE YOUR REFLECTION HERE]
```

#### Simulated Collaboration (Step 5)

Paste the output of the analysis cell (the comparison table and the realism summary).

```
[PASTE THE OUTPUT HERE]
```

An excerpt (2-3 turns) where the simulated user is convincingly you, and one where it breaks down. Include the session ids.

```
[CONVINCING EXCERPT, SESSION ID]
```

```
[BREAKDOWN EXCERPT, SESSION ID]
```

How realistic was your simulated self, and where did it break down? Did Coco behave differently with the simulated user than with you in Question 1.1? Would you trust simulated trajectories like these to evaluate a collaborative agent? (4-6 sentences)

```
[WRITE YOUR REFLECTION HERE]
```

## Building Automatic Evaluators to Approximate Human Judgement (50 points + 10 extra credit)

### Question 2.1 (10 points)

Copy and paste your HelpSteer2 prompt and Pearson correlation here.

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

And your Pearson Correlation on HelpSteer2

Pearson correlation: X.XXX ± X.XXX

### Question 2.4 (20 points + 10 extra credit for top 3 submissions)

Paste your implementation of `generate_evaluator` here.

```python
# Given a train_df and human_criteria, generate an automatic evaluator that can be used to score a model's output
# Inputs:
#   train_df: a dataframe of training examples with input, output, and score columns
#   human_criteria: a string of the human criteria for the task (e.g. "helpfulness", "simplification quality")
# Outputs:
#   A callable function that takes an input and output and returns a score
def generate_evaluator(train_df, human_criteria="Unknown") -> Callable[[str, str], float]:
    # TODO: Replace this implementation with your code here!  This implements the generation + regression method from above
```

Paste your results on HelpSteer2 and SimpEval below.

```
[HelpSteer2] Pearson correlation:  X.XXX ± X.XXX
[SimpEval] Pearson correlation:  X.XXX ± X.XXX
Average Pearson correlation:  X.XXX
```

Finally please write up a 1-2 paragraph explanation of your approach.  Be sure to cite any papers that introduce similar methods if you took inspiration from the literature.

```
[WRITE YOUR 1-2 PARAGRAPH EXPLANATION OF YOUR APPROACH HERE]
```


### Question 2.5 (X points)

Paste the criteria and weights from the criteria cell (the top 5 are enough) and the Pearson correlation on the test sessions.

```
[CRITERIA AND WEIGHTS]
```

Pearson correlation on the test sessions: X.XXX

Do the learned criteria match what mattered to you when you rated Coco? On which sessions does the evaluator disagree with you most, and why might that be? How does this compare to your results on HelpSteer2 and SimpEval in Question 2.4? Would you use this evaluator to compare two versions of Coco, and what would make it more trustworthy? (4-6 sentences)

```
[WRITE YOUR REFLECTION HERE]
```
