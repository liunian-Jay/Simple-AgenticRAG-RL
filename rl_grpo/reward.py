import re
from math_verify import parse, verify, ExprExtractionConfig

import string
from collections import Counter
  

#################################################
########        READER EVALUATION        ########
#################################################
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
  """ EM: ground_truth is a str """
  return normalize_answer(prediction) == normalize_answer(ground_truth)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            normalize_answer(pattern),
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(normalize_answer(text)) is not None
    
def f1_score(prediction, ground_truth):
  """ F1: ground_truth is a str """
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
      return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def em_max_over_ground_truths(prediction, ground_truths, regex=True):
  """ EM: ground_truth is a list """
  return max([regex_match(prediction, gt) if regex else exact_match_score(prediction, gt) for gt in ground_truths])

def f1_max_over_ground_truths(prediction, ground_truths):
  """ F1: ground_truth is a list """
  return max([f1_score(prediction, gt) for gt in ground_truths])


def reward_EM_F1(item, res_answer):
    """ 供直接使用 """
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, res_answer)
    if match:
        answer_text = match.group(1).strip()
        em_score = em_max_over_ground_truths(answer_text, item['answers'], regex = True)
        f1_score = f1_max_over_ground_truths(answer_text, item['answers'])
        return (em_score + f1_score)/2 + 1
    return -1.0 # 相当于格式正确奖励0，错误奖励-1


###############################################
###############################################
###############################################
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0: return -1.0
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

def reward_format(item, answer):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = answer.count("<think>") + answer.count("</think>")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else -1
