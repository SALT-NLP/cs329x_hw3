# CS329X HW3: Building and Evaluating Human-AI Interaction

**Due 11/19/25 11:59PM PT**

75 points total + 10 points extra credit

## Overview

In this assignment, you will implement methods from two recent papers which focus on builing or evaluating human-ai interaction:
1. **CoGym** - a framework for building and evaluating collaborative agents that take initiative at the correct time and work together with humans in parallel.
2. **AutoMetrics** - a library for generating automatic evaluators to approximate human-judgements


You will also have an opportunity to recieve extra credit by competing on a leaderboard to build the best automatic evaluators.

This assignment will require that you use your **Google Cloud Credits**.  We estimate that this assignment will take fewer than $20.  If you do not have enough credit remaining please contact the teaching team.


## File Structure

### Core Files

  - **`hw3.ipynb`** - Main Jupyter notebook containing the complete assignment workflow. This is where you'll implement most of your code and run experiments.
  - **`writeup.md`** - Template for your written responses and analysis. You must fill this out with your answers.

### Environment

  - **`.env.example`** - example of how you need to structure your `.env` file
  - **`.env`** - file you will create to store your GEMINI_API_KEY
  - **`requirements.txt`** - file specifying the library requirements that you need to install for the assignment

### Leaderboard

  - **`submission.py`** - file to modify with your custom `generate_evaluator` method to submit to the leaderboard.

## Environment Setup

1. Install required packages (we suggest you install them in Conda environment):
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in the API keys.

3. Open and run through `hw3.ipynb` following the instructions

## Submission Requirements

### What to Submit

You must submit a **ZIP file** containing:

1. **All code files** including:
   - `hw3.ipynb` (with all cells executed and outputs visible)
   - `submission.py` for submitting to the leaderboard
   - `.env` so that we can run your submission on our private test set

2. **Completed writeup**:
   - Export `writeup.md` with all TODO sections filled out as a PDF file. 

For submitting to the **gradescope leaderboard** you may find `submit.sh` useful for zipping your files for submission.  For **canvas upload** please be additionally sure to convert `writeup.md` to a PDF file.

**This is the first homework that will involve gradescope submission, so please be careful to remember to upload to BOTH gradescope and canvas**.

## Important Notes

- Test that your code runs before submission
- Include all outputs in your Jupyter notebook submission
- Double-check that all TODO sections in `writeup.md` are completed

Good luck building and evaluating human-ai interaction!
