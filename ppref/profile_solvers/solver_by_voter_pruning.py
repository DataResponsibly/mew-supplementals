from collections import Counter
from functools import lru_cache
from math import isclose, ceil
from multiprocessing import cpu_count, Pool
from random import random
from time import time

from ppref.models.mallows import Mallows
from ppref.models.rim import RepeatedInsertionModel
from ppref.models.rsm import RepeatedSelectionModelRV
from ppref.preferences.combined import MallowsWithPP, RimWithTR, MallowsWithPoset
from ppref.preferences.poset import Poset
from ppref.preferences.special import PartitionedWithMissing, PartitionedPreferences, PartialChain, TruncatedRanking
from ppref.profile_solvers.posets.helper import prune_candidates_w_too_low_upper_bound
from ppref.rank_estimators.mallows_w_pp import calculate_rank_probs_for_item_given_mallows_w_pp
from ppref.rank_estimators.rim import calculate_rank_probs_for_item_given_rim
from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_sequential
from ppref.rank_estimators.rim_w_trun import calculate_rank_probs_for_item_given_rim_w_trun
from ppref.rank_estimators.rsm import calculate_rank_prob_for_item_given_rsm


def quickly_compute_upper_lower_bounds_weighted(vote_counts: list[tuple], rule_vector: tuple[int]):
    candidates = vote_counts[0][0].get_full_item_set()
    candidate2upper = {c: 0.0 for c in candidates}
    candidate2lower = {c: 0.0 for c in candidates}

    # quickly calculate upper and lower bounds

    for pref, weight in vote_counts:
        for c in candidates:
            rank_left, rank_right = pref.get_range_of_possible_ranks(c)
            candidate2upper[c] += rule_vector[rank_left] * weight
            candidate2lower[c] += rule_vector[rank_right] * weight

    return candidate2upper, candidate2lower


@lru_cache(maxsize=100)
def compute_poset_posterior(mallows: Mallows, poset: Poset):
    item = mallows.reference[-1]
    answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=item, rim=mallows, poset=poset)
    return sum(answer['probabilities'])


def general_abstract_solver(pref, item, max_rank=None):
    if isinstance(pref, Poset):
        if pref.has_item(item):
            mallows = Mallows(reference=pref.item_set_tuple, phi=1.0)
            posterior = compute_poset_posterior(mallows, pref)
            answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=item, rim=mallows, poset=pref,
                                                                                   max_rank=max_rank)
            return [prob / posterior for prob in answer['probabilities']]
        else:
            return [1 / len(pref.item_set) for _ in range(max_rank + 1)]
    elif isinstance(pref, PartitionedWithMissing):
        return pref.calculate_rank2prob_for_item(item)
    elif isinstance(pref, PartitionedPreferences):
        return pref.calculate_rank_probabilities_of_item(item)
    elif isinstance(pref, PartialChain):
        return pref.calculate_rank2prob_for_item(item)
    elif isinstance(pref, TruncatedRanking):
        return pref.calculate_rank_probs_for_item(item)
    elif isinstance(pref, RepeatedInsertionModel):
        return calculate_rank_probs_for_item_given_rim(item, pref, max_rank)
    elif isinstance(pref, RepeatedSelectionModelRV):
        return calculate_rank_prob_for_item_given_rsm(item, pref, max_rank)
    elif isinstance(pref, MallowsWithPP):
        return calculate_rank_probs_for_item_given_mallows_w_pp(item, pref, max_rank)
    elif isinstance(pref, RimWithTR):
        return calculate_rank_probs_for_item_given_rim_w_trun(item, pref, max_rank)
    elif isinstance(pref, MallowsWithPoset):
        mallows = pref.mallows
        poset = pref.poset
        posterior = compute_poset_posterior(mallows, poset)
        answer = calculate_rank_probs_for_item_given_rim_w_poset_by_sequential(item=item, rim=mallows, poset=poset,
                                                                               max_rank=max_rank)
        return [prob / posterior for prob in answer['probabilities']]


def sequential_solver_of_voter_pruning(profile: list, rule_vector: tuple[int], verbose=False, grouping=True):
    max_rank = rule_vector.index(0) - 1

    t_pruning_sec = 0
    t_solver_sec = 0

    # pre-process
    vote_counts = [(pref, 1) for pref in profile]

    t0 = time()
    if grouping:
        vote_counts = sorted(Counter(profile).items(), key=lambda x: x[1], reverse=True)

    t1 = time()
    candidate2upper, candidate2lower = quickly_compute_upper_lower_bounds_weighted(vote_counts, rule_vector)
    t2 = time()
    candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)
    t3 = time()

    t_pruning_sec += t3 - t2

    if len(candidate2upper) == 1:
        winner, upper = candidate2upper.popitem()
        lower = candidate2lower[winner]
        return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower, 'num_pruned_voters': len(profile),
                't_vote_count_sec': t1 - t0, 't_quick_bounds_sec': t2 - t1, 't_pruning_sec': t_pruning_sec,
                't_solver_sec': 0, 't_total_sec': t3 - t0}

    num_processed_voters = 0
    while vote_counts:
        pref, weight = vote_counts.pop()
        is_refined = False

        if verbose:
            num_processed_voters += weight
            if random() < 0.1:
                print(f'[INFO] Processing voters {num_processed_voters}')

        for c in candidate2upper:
            rank_left, rank_right = pref.get_range_of_possible_ranks(c)
            score_left, score_right = rule_vector[rank_left], rule_vector[rank_right]
            if not isclose(score_left, score_right):
                t4 = time()
                probs = general_abstract_solver(pref, c, max_rank)
                t5 = time()
                t_solver_sec += t5 - t4

                exact_score = sum([prob * val for prob, val in zip(probs[:max_rank + 1], rule_vector[: max_rank + 1])])
                candidate2upper[c] += (exact_score - score_left) * weight
                candidate2lower[c] += (exact_score - score_right) * weight
                is_refined = True

        if is_refined:
            t6 = time()
            candidate2upper, candidate2lower = prune_candidates_w_too_low_upper_bound(candidate2upper, candidate2lower)
            t7 = time()
            t_pruning_sec += t7 - t6

            if len(candidate2upper) == 1:
                t8 = time()
                winner, upper = candidate2upper.popitem()
                lower = candidate2lower[winner]
                num_pruned_voters = sum([w for _, w in vote_counts])
                return {'winners': (winner,), 'score_upper': upper, 'score_lower': lower,
                        'num_pruned_voters': num_pruned_voters, 't_vote_count_sec': t1 - t0, 't_quick_bounds_sec': t2 - t1,
                        't_pruning_sec': t_pruning_sec, 't_solver_sec': t_solver_sec, 't_total_sec': t8 - t0}

    winner_score = max(candidate2upper.values())
    winners = tuple(candidate2upper.keys())
    t9 = time()
    return {'winners': winners, 'score_upper': winner_score, 'score_lower': winner_score, 'num_pruned_voters': 0,
            't_vote_count_sec': t1 - t0, 't_quick_bounds_sec': t2 - t1, 't_pruning_sec': t_pruning_sec,
            't_solver_sec': t_solver_sec, 't_total_sec': t9 - t1}


def compute_candidate_scores(vote_counts: list, rule_vector: tuple[int], verbose=False):
    t0 = time()
    max_rank = rule_vector.index(0) - 1
    candidates = vote_counts[0][0].get_full_item_set()
    candidate2score = {c: 0.0 for c in candidates}

    for pref, weight in vote_counts:
        for c in candidates:
            probs = general_abstract_solver(pref, c, max_rank)
            exact_score = sum([prob * val for prob, val in zip(probs[:max_rank + 1], rule_vector[: max_rank + 1])])
            candidate2score[c] += exact_score * weight

    if verbose:
        print(f'  -> [INFO] compute_candidate_scores: {len(vote_counts)=}, time = {time() - t0: .3f} s')

    return candidate2score


def compute_winners_from_candidate_scores(candidate2score: dict):
    winner_score = max(candidate2score.values())
    winners = tuple([c for c, s in candidate2score.items() if s == winner_score])
    return winners, winner_score


def sequential_baseline_solver(profile: list, rule_vector: tuple[int], grouping=False):
    """No pruning. Just by computing scores of all candidates."""

    # pre-process
    vote_counts = [(pref, 1) for pref in profile]

    t1 = time()
    if grouping:
        vote_counts = sorted(Counter(profile).items(), key=lambda x: x[1], reverse=True)

    candidate2score = compute_candidate_scores(vote_counts, rule_vector)
    winners, winner_score = compute_winners_from_candidate_scores(candidate2score)
    t2 = time()
    return {'winners': winners, 'winner_score': winner_score, 't_sec': t2 - t1}


def parallel_baseline_solver(profile: list, rule_vector: tuple[int], threads=None, grouping=True):
    threads = threads or cpu_count() // 2

    if threads == 1:
        return sequential_baseline_solver(profile, rule_vector, grouping)

    vote_counts = [(pref, 1) for pref in profile]

    # pre-process
    t0 = time()
    if grouping:
        vote_counts = sorted(Counter(profile).items(), key=lambda x: x[1], reverse=True)

    t1 = time()
    chunksize = ceil(len(vote_counts) / threads)
    tasks = []
    for i in range(0, len(vote_counts), chunksize):
        tasks.append((vote_counts[i:i + chunksize], rule_vector))

    with Pool(threads) as pool:
        candidate2score_list = pool.starmap(compute_candidate_scores, tasks)

    candidate2score = candidate2score_list.pop()
    for c2s in candidate2score_list:
        for c, s in c2s.items():
            candidate2score[c] += s

    winners, winner_score = compute_winners_from_candidate_scores(candidate2score)

    t4 = time()
    return {'winners': winners, 'winner_score': winner_score, 't_sec': t4 - t0}


def main():
    from experiments.synthetic.posets.experiment import generate_profile_df

    k_approval, num_candidates, num_voters, phi, rsm_pmax, batch = 5, 10, 1_000, 0.5, 0.1, 8
    thread_list = [2, 3, 4, 6]

    rule_vector = tuple([1 for _ in range(k_approval)] + [0 for _ in range(num_candidates - k_approval)])

    t0 = time()
    df = generate_profile_df(num_candidates, num_voters, phi, rsm_pmax, batch)
    profile = df['poset'].tolist()
    t_profile_gen = time() - t0
    print(f'[INFO] {t_profile_gen=:.3f} s')
    print(sequential_solver_of_voter_pruning(profile.copy(), rule_vector=rule_vector))

    for threads in thread_list:
        print(parallel_baseline_solver(profile.copy(), rule_vector=rule_vector, threads=threads))


if __name__ == '__main__':
    main()
