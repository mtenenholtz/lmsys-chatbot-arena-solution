from typing import List, Dict, Optional, Any
from datasets import load_dataset
import random
random.seed(42)
import polars as pl

# Load the dataset
dataset = load_dataset("openbmb/UltraFeedback", split="train")#test it: .select(range(10))


def calculate_average_rating(annotations: Dict[str, Any]) -> Optional[float]:
    ratings = [int(details['Rating']) for details in annotations.values() if 'Rating' in details and details['Rating'] != "N/A"]
    return sum(ratings) / len(ratings) if ratings else None

def select_rejected_responses(completions: List[Dict[str, Any]], comparison_key: str, best_score: float) -> Optional[Dict[str, Any]]:
    eligible_responses = [resp for resp in completions if resp.get(comparison_key) is not None]
    return random.choice(eligible_responses) if eligible_responses else None

def process_dataset(record: Dict[str, Any]) -> Dict[str, Any]:
    completions = record.get('completions', [])

    if not completions:
        return {**record, 'best_rated_response': None, 'random_response_for_rated': None}

    for response in completions:
        response['average_rating'] = calculate_average_rating(response.get('annotations', {}))

    best_rated_response = max(completions, key=lambda x: x.get('average_rating', -1))

    # rejected_responses_list = select_rejected_responses(completions, 'average_rating', best_rated_response.get('average_rating', -1))
    # rejected_ratings = []
    # rejected_responses = []
    # rejected_models = []
    # for rejected in rejected_responses_list:
    #     rejected_ratings.append(rejected['average_rating'])
    #     rejected_responses.append(rejected['response'])
    #     rejected_models.append(rejected['model'])

    rejected_response = select_rejected_responses(completions, 'average_rating', best_rated_response.get('average_rating', -1))

    if random.random() < 0.5:
        return {
            'prompt': [record['instruction']],
            'response_a': [best_rated_response['response']],
            'response_b': [rejected_response['response']],
            'fold': -1
        }
    else:
        return {
            'prompt': [record['instruction']],
            'response_a': [rejected_response['response']],
            'response_b': [best_rated_response['response']],
            'fold': -1
        }

results = [process_dataset(record) for record in dataset if len(record['completions'])>0]
df = pl.from_records(results)
df.write_parquet('data/ultrafeedback.parquet')