# CS329X HW3: Building and Evaluating Human-AI Interaction

**Due 11/11/26 11:59PM PT**

75 points total + 10 points extra credit

## Overview

In this assignment, you will build on recent work on building and evaluating human-ai interaction:
1. **Coco** - an open-source collaborative agent that you will use to collect your own human-agent collaboration traces.
2. **AutoMetrics** - a library for generating automatic evaluators to approximate human-judgements


You will also have an opportunity to recieve extra credit by competing on a leaderboard to build the best automatic evaluators.

This assignment will require that you use your **Google Cloud Credits**.  We estimate that this assignment will take fewer than $20.  If you do not have enough credit remaining please contact the teaching team.


## File Structure

### Core Files

  - **`hw3.ipynb`** - Main Jupyter notebook containing the complete assignment workflow. This is where you'll implement most of your code and run experiments.
  - **`writeup.md`** - Template for your written responses and analysis. You must fill this out with your answers.
  - **`export_coco_traces.py`** - script to export your Coco chat sessions to `traces/coco_traces.json`.

### Environment

  - **`.env.example`** - example of how you need to structure your `.env` file
  - **`.env`** - file you will create to store your GEMINI_API_KEY and your Persimmon API key (PERSIMMON_API_KEY)
  - **`requirements.txt`** - file specifying the library requirements that you need to install for the assignment

### Leaderboard

  - **`submission.py`** - file to modify with your custom `generate_evaluator` method to submit to the leaderboard.

## Environment Setup

1. Create a virtual environment with [uv](https://docs.astral.sh/uv/) and install the required packages:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in the API keys.

3. Open and run through `hw3.ipynb` following the instructions, using the `.venv` environment as the notebook kernel

## Submission Requirements

### What to Submit

You must submit a **ZIP file** containing:

1. **All code files** including:
   - `hw3.ipynb` (with all cells executed and outputs visible)
   - `submission.py` for submitting to the leaderboard
   - `traces/coco_traces.json` with the collaboration traces you collected in Part 1
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
