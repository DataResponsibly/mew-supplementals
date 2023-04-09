import operator
from collections import defaultdict
from functools import partial
from math import ceil
from multiprocessing import cpu_count, Pool
from random import random
from time import time

from ppref.helper import is_running_out_of_memory


def tuple_adder(a: tuple[int], b: tuple[int]):
    return tuple(map(operator.add, a, b))


def compute_new_outcome(outcome, ranking, rule_vector, candidate_list, candidate2cid):
    current_scores = [0 for _ in candidate_list]
    for cand, score in zip(ranking, rule_vector):
        if score == 0:
            break
        else:
            cid = candidate2cid[cand]
            current_scores[cid] += score

    outcome_new = tuple_adder(outcome, tuple(current_scores))

    return outcome_new


def determine_winners_by_outcomes(outcome2count, candidate_list):
    candidate_to_winner_count = defaultdict(int)
    while outcome2count:
        outcome, count = outcome2count.popitem()

        winning_score = max(outcome)
        for idx, score in enumerate(outcome):
            if score == winning_score:
                winner = candidate_list[idx]
                candidate_to_winner_count[winner] += count

    winners = set()
    num_winning_possible_worlds = max(candidate_to_winner_count.values())
    for cand, count in candidate_to_winner_count.items():
        if count == num_winning_possible_worlds:
            winners.add(cand)

    return winners, num_winning_possible_worlds


def sequential_mpw_solver(profile: list, rule_vector: tuple[int], verbose=False, is_improved=False):
    """
    It only works for profile: list[Poset].
    """
    candidate_list = list(profile[0].get_full_item_set())
    candidate2cid = {cand: idx for idx, cand in enumerate(candidate_list)}

    outcome2count, outcome2count_temp = defaultdict(int), defaultdict(int)
    outcome2count[tuple([0 for _ in candidate_list])] = 1

    if verbose:
        print(f'[INFO] {candidate_list=} \n[INFO] Compiling the profile...')

    t1 = time()
    voter_counter = 0
    for pref in profile:
        voter_counter += 1

        t_round_start = time()

        if is_improved:
            for ranking in pref.iterate_linear_extensions():
                for outcome, count in outcome2count.items():
                    outcome_new = compute_new_outcome(outcome, ranking, rule_vector, candidate_list, candidate2cid)
                    outcome2count_temp[outcome_new] += count
        else:
            for outcome, count in outcome2count.items():
                for ranking in pref.iterate_linear_extensions():
                    outcome_new = compute_new_outcome(outcome, ranking, rule_vector, candidate_list, candidate2cid)
                    outcome2count_temp[outcome_new] += count

        outcome2count = outcome2count_temp.copy()
        outcome2count_temp.clear()

        if verbose:
            print(f'[INFO] Finish voter {voter_counter}, time = {time() - t_round_start:.3f} s, size = {len(outcome2count)}')

    t2 = time()

    num_outcomes = len(outcome2count)
    num_possible_worlds = sum(outcome2count.values())

    winners, num_winning_possible_worlds = determine_winners_by_outcomes(outcome2count, candidate_list)

    t3 = time()

    return {'winners': winners, 'winning_probability': num_winning_possible_worlds / num_possible_worlds,
            'num_outcomes': num_outcomes, 't_compile_sec': t2 - t1, 't_post_compile_sec': t3 - t2, 't_total_sec': t3 - t1}


def worker(pref, candidate_list, candidate2cid, rule_vector, t_max_min, is_improved, outcome_count_pairs: list[tuple, int]):
    t0 = time()
    t_max_s = t_max_min * 60

    outcome2count_temp = defaultdict(int)

    if is_improved:
        for ranking in pref.iterate_linear_extensions():
            for outcome, count in outcome_count_pairs:

                if random() < 0.00001:
                    if is_running_out_of_memory(verbose=False):
                        return {'has_answer': False, 'error': 'out-of-memory'}
                    elif time() - t0 > t_max_s:
                        return {'has_answer': False, 'error': f'unfinished-in-{t_max_min}-min'}

                outcome_new = compute_new_outcome(outcome, ranking, rule_vector, candidate_list, candidate2cid)
                outcome2count_temp[outcome_new] += count
    else:
        for outcome, count in outcome_count_pairs:
            for ranking in pref.iterate_linear_extensions():

                if random() < 0.00001:
                    if is_running_out_of_memory(verbose=False):
                        return {'has_answer': False, 'error': 'out-of-memory'}
                    elif time() - t0 > t_max_s:
                        return {'has_answer': False, 'error': f'unfinished-in-{t_max_min}-min'}

                outcome_new = compute_new_outcome(outcome, ranking, rule_vector, candidate_list, candidate2cid)
                outcome2count_temp[outcome_new] += count

    return {'has_answer': True, 'answer': outcome2count_temp}


def parallel_mpw_solver(profile: list, rule_vector: tuple[int], threads: int = None, t_max_min=30, verbose=False,
                        is_improved=False):
    """
    It only works for profile: list[Poset].
    """
    threads = threads or cpu_count()
    t_max_s = t_max_min * 60

    candidate_list = list(profile[0].get_full_item_set())
    candidate2cid = {cand: idx for idx, cand in enumerate(candidate_list)}

    outcome2count, outcome2count_temp = defaultdict(int), defaultdict(int)
    outcome2count[tuple([0 for _ in candidate_list])] = 1

    if verbose:
        print(f'[INFO] {candidate_list=} \n[INFO] Compiling the profile...')

    t1 = time()
    voter_counter = 0
    for pref in profile:
        voter_counter += 1

        t_round_start = time()
        if t_round_start - t1 > t_max_s:
            return {'has_answer': False, 'error': f'unfinished-in-{t_max_min}-min'}

        worker_partial = partial(worker, pref, candidate_list, candidate2cid, rule_vector, t_max_min, is_improved)
        single_worker_size = max(1, ceil(len(outcome2count) / threads))

        batches = []
        for _ in range(threads):
            batch = [outcome2count.popitem() for _ in range(min(single_worker_size, len(outcome2count)))]
            batches.append(batch)

        with Pool(threads) as pool:
            try:
                all_res = pool.map(worker_partial, batches)
            except Exception as e:
                pool.terminate()
                print(f'[INFO] OutOfMemory, {e.__class__}')
                return {'has_answer': False, 'error': 'out-of-memory'}

        if {'has_answer': False, 'error': 'out-of-memory'} in all_res:
            return {'has_answer': False, 'error': 'out-of-memory'}
        elif {'has_answer': False, 'error': f'unfinished-in-{t_max_min}-min'} in all_res:
            return {'has_answer': False, 'error': f'unfinished-in-{t_max_min}-min'}
        else:
            outcome2count_temp = all_res.pop()['answer']
            while all_res:
                answer = all_res.pop()['answer']
                while answer:
                    outcome, count = answer.popitem()
                    outcome2count_temp[outcome] += count

        outcome2count, outcome2count_temp = outcome2count_temp, outcome2count

        if verbose:
            print(f'[INFO] Finish voter {voter_counter}, time = {time() - t_round_start:.3f} s, size = {len(outcome2count)}')

    t2 = time()

    num_outcomes = len(outcome2count)
    num_possible_worlds = sum(outcome2count.values())

    winners, num_winning_possible_worlds = determine_winners_by_outcomes(outcome2count, candidate_list)

    t3 = time()

    return {'has_answer': True, 'winners': winners, 'winning_probability': num_winning_possible_worlds / num_possible_worlds,
            'num_outcomes': num_outcomes, 't_compile_sec': t2 - t1, 't_post_compile_sec': t3 - t2, 't_total_sec': t3 - t1}


def main():
    from experiments.synthetic.posets.experiment import generate_profile_df
    from experiments.helper import get_rule_vector

    num_candidates, num_voters, phi, rsm_pmax, batch = 4, 5, 0.5, 0.1, 0

    print(f'[INFO] Generating a profile of {num_voters} voters...')
    t0 = time()
    df = generate_profile_df(num_candidates, num_voters, phi, rsm_pmax, batch)
    print(f'[INFO] Profile is ready. ({time() - t0} seconds.)')
    posets = df['poset'].tolist()
    rule_vector = get_rule_vector('Borda', num_candidates)

    print(sequential_mpw_solver(posets.copy(), rule_vector=rule_vector, verbose=True))
    print(sequential_mpw_solver(posets.copy(), rule_vector=rule_vector, verbose=True, is_improved=True))
    # print(parallel_mpw_solver(posets.copy(), rule_vector=rule_vector, verbose=True))


if __name__ == '__main__':
    main()
