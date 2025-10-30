import pandas as pd
from IPython.display import display
from dotenv import load_dotenv
import litellm
from litellm import disable_cache, enable_cache
from litellm.caching.caching import Cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Callable, Optional

litellm.cache = Cache(type="disk")

litellm.set_verbose = False
litellm.suppress_debug_info = True

pd.set_option("display.max_colwidth", None)   # show full text in each cell
pd.set_option("display.width", 2000)          # prevent horizontal truncation
pd.set_option("display.max_columns", None)    # show all columns

load_dotenv()

def llm(prompt="", model="gemini/gemini-flash-latest"):
    if type(prompt) == str:
        prompt = [{"role": "user", "content": prompt}]

    return litellm.completion(
        model=model,
        messages=prompt,
    ).choices[0].message.content


def _llm_as_judge_prompt(prompt, model="gemini/gemini-flash-latest"):
    system_prompt = """You are a expert evaluator.  Given the criteria that you are evaluating for, you will score the given response from 1-5.

First, reason over the given response and how it does or does not meet the criteria.  Then, give your final score.

Follow the following format:

<Reasoning>
[Your reasoning which will help you come up with your score]
</Reasoning>
<Score>
[Your final score; 1-5]"""

    if type(prompt) == str:
        prompt = [{"role": "user", "content": prompt}]

    prompt = [{"role": "system", "content": system_prompt}] + prompt
    
    try:
        response = llm(prompt, model)

        # Extract score from response (the next line after <Score>)
        score_line = response.split("<Score>")[1].split("</Score>")[0].strip()
        score = float(score_line)
        
        return score
    except Exception as e:
        prompt[-1]["content"] = prompt[-1]["content"] + "\n\nBe very careful and precise in formatting your response."
        res2 = llm(prompt)
        score_line = res2.split("<Score>")[1].split("</Score>")[0].strip()
        score = float(score_line)
        
        return score

def llm_as_judge(evaluation_criteria, input, output, model="gemini/gemini-flash-latest"):
    prompt = f"""<Evaluation Criteria>
    {evaluation_criteria}
    </Evaluation Criteria>
    <Input provided to the AI>
    {input}
    </Input>
    <Output to evaluate>
    {output}
    </Output>
    """
    return _llm_as_judge_prompt(prompt, model)


def run_llm_as_judge_on_df(df, evaluation_criteria, model="gemini/gemini-flash-latest", num_threads=16):
    scores = [None] * len(df)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(
            llm_as_judge,
            evaluation_criteria,
            row.input,
            row.output,
            model,
        ): i for i, row in enumerate(df.itertuples(index=False))}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring rows"):
            i = futures[future]
            scores[i] = future.result()
    return scores

def run_regression_judge_on_df(df, criteria_list, coef, intercept, model="gemini/gemini-flash-latest", num_threads=16):
    for criterion in criteria_list:
        scores = run_llm_as_judge_on_df(df, criterion, model, num_threads)
        df[criterion] = scores

    coef_arr = np.asarray(coef).reshape(-1)  # force (n_features,)

    df['predicted_score'] = df[criteria_list].dot(coef_arr) + intercept

    return df['predicted_score'].tolist()


def run_evaluator_on_df(df, evaluator: Callable[[str, str], float], num_threads=16) -> list[float]:
    # Preallocate result list
    scores = [None] * len(df)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit one task per row
        futures = {
            executor.submit(evaluator, row.input, row.output): i
            for i, row in enumerate(df.itertuples(index=False))
        }

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring rows"):
            i = futures[future]
            scores[i] = future.result()

    return scores

def compute_pearson_correlation(df, scores: list[float]):
    scores_series = pd.Series(scores, index=df.index, name="predicted_score")

    return df["score"].corr(scores_series, method="pearson")

def get_judge_confidence_interval(df, evaluation_criteria, model="gemini/gemini-flash-latest", num_threads=16, trials=5):
    disable_cache()
    scores = []
    for _ in range(trials):
        print(f"Running trial {_ + 1} of {trials}")
        scores.append(run_llm_as_judge_on_df(df, evaluation_criteria, model, num_threads))
    enable_cache()

    pearson_correlations = []
    for score_list in scores:
        pearson_correlations.append(compute_pearson_correlation(df, score_list))
    return np.array(pearson_correlations).mean(), np.array(pearson_correlations).std() / np.sqrt(trials)

def get_regression_confidence_interval(df, criteria_list, coef, intercept, model="gemini/gemini-flash-latest", num_threads=16, trials=5):
    disable_cache()
    scores = []
    for _ in range(trials):
        print(f"Running trial {_ + 1} of {trials}")
        scores.append(run_regression_judge_on_df(df, criteria_list, coef, intercept, model, num_threads))
    enable_cache()

    pearson_correlations = []
    for score_list in scores:
        pearson_correlations.append(compute_pearson_correlation(df, score_list))
    return np.array(pearson_correlations).mean(), np.array(pearson_correlations).std() / np.sqrt(trials)

def get_evaluator_confidence_interval(df, evaluator:Callable[[str, str], float], num_threads=16, trials=5) -> tuple[float, float]:
    disable_cache()
    scores = []
    for _ in range(trials):
        scores.append(run_evaluator_on_df(df, evaluator, num_threads))
    enable_cache()

    pearson_correlations = []
    for score_list in scores:
        pearson_correlations.append(compute_pearson_correlation(df, score_list))
    return np.array(pearson_correlations).mean(), np.array(pearson_correlations).std() / np.sqrt(trials)

def evaluate_on_df_set(
    train_df_list,
    test_df_list,
    evaluator_constructor: Callable[[pd.DataFrame, str], Callable[[str, str], float]],
    human_criteria_list: Optional[list[str]] = None,
    num_threads: int = 16,
    trials: int = 5,
    parallel: bool = False,
):
    if human_criteria_list is None:
        human_criteria_list = ["Unknown"] * len(train_df_list)

    # package args per task
    task_args = list(zip(train_df_list, test_df_list, human_criteria_list))

    def run_one(train_df, test_df, human_criteria):
        evaluator = evaluator_constructor(train_df, human_criteria)
        mean, std = get_evaluator_confidence_interval(
            test_df,
            evaluator,
            num_threads=num_threads,
            trials=trials,
        )
        return (mean, std)

    if parallel:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            results = list(ex.map(lambda args: run_one(*args), task_args))
    else:
        results = [run_one(*args) for args in task_args]

    # side-effect print, same as before
    for mean, std in results:
        print(f"Pearson correlation: {mean} ± {std}")

    return results

class Dataset():
    def __init__(self, train_df:pd.DataFrame, dev_df:pd.DataFrame, test_df:pd.DataFrame, human_criteria:str="Unknown"):
        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df
        self.human_criteria = human_criteria
        
def init_dataset(path:str) -> Dataset:
    train_df = pd.read_csv(f"{path}/train.csv")
    dev_df = pd.read_csv(f"{path}/val.csv")
    test_df = pd.read_csv(f"{path}/test.csv")
    with open(f"{path}/human_criteria.txt", "r") as f:
        human_criteria = f.read().strip()
    return Dataset(train_df, dev_df, test_df, human_criteria)

def evaluate_on_datasets(datasets:list[Dataset], evaluator_constructor:Callable[[pd.DataFrame, str], Callable[[str, str], float]], num_threads=16, trials=5, parallel=False):
    return evaluate_on_df_set(
        [dataset.train_df for dataset in datasets],
        [dataset.test_df for dataset in datasets],
        evaluator_constructor,
        [dataset.human_criteria for dataset in datasets],
        num_threads,
        trials,
        parallel
    )