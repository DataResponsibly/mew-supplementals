from multiprocessing import Pool

import pandas as pd
from numpy.random import default_rng

from experiments.helper import get_dir, get_random_poset_by_rsm, get_rule_vector
from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning


def generate_combs():
    m = 10

    n_list = [100, 1000, 10_000, 100_000]
    phi_list = [0.1, 0.5, 0.9]
    batch_list = list(range(10))

    combs = set()

    # varying phi
    pmax, n = 0.1, 1000
    for phi in phi_list:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    # varying n
    pmax, phi = 0.1, 0.5
    for n in n_list:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    return list(combs)


def generate_profile_df(m, n, phi, pmax, batch, verbose=False):
    rng = default_rng([m, n, int(phi * 1000), int(pmax * 1000), batch])

    mallows = Mallows(reference=tuple(range(m)), phi=phi)
    df = pd.DataFrame(columns=['m', 'phi', 'pmax', 'cardinality', 'poset'])
    for i in range(n):
        if verbose and i % (n // 10) == 0:
            print(f'[INFO] generate_profile_df({n=},...) is generating voter {i}')
        ranking = mallows.sample_a_ranking(rng)
        probs = rng.random(size=m) * pmax
        poset = get_random_poset_by_rsm(ranking, probs, rng)
        cardinality = poset.dag.number_of_nodes()

        df.loc[i] = [m, phi, pmax, cardinality, poset]

    return df


def generate_profiles():
    for (m, n, phi, pmax, batch) in generate_combs():
        filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
        fullpath = get_dir(__file__) / f'profiles/{filename}'
        fullpath.parent.mkdir(parents=True, exist_ok=True)

        if not fullpath.exists():
            print(f'[INFO] Generating {filename}')
            df = generate_profile_df(m, n, phi, pmax, batch)
            df.to_csv(fullpath, index=False, sep='\t')

    print('Profile Generation... Done.')


def run_experiment():
    sep = '\t'

    threads = 1
    pruning = True
    grouping = True

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'threads', 'pruning', 'grouping', 'winners', 'score_upper',
                    'score_lower', 'num_pruned_voters', 't_vote_count_sec', 't_quick_bounds_sec', 't_pruning_sec',
                    't_solver_sec', 't_total_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, phi, pmax, batch) in generate_combs():
            for rule in rule_list:

                condition = f'(m == {m}) and (n == {n}) and (phi == {phi}) and (pmax == {pmax}) and (batch == {batch}) ' \
                            f'and (rule == "{rule}") and (threads == {threads}) and (grouping == {grouping})'

                is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                if not is_existing:
                    print(f'- Executing {condition}')

                    filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
                    fullpath = get_dir(__file__) / f'profiles/{filename}'
                    df_in = pd.read_csv(fullpath, sep=sep)

                    profile: list[Poset] = [eval(po) for po in df_in['poset']]
                    answer = sequential_solver_of_voter_pruning(profile, get_rule_vector(rule, m), grouping=grouping)

                    record = [m, n, f'{phi:.1f}', f'{pmax:.1f}', batch, rule, threads, pruning, grouping, answer['winners'],
                              answer['score_upper'], answer['score_lower'], answer['num_pruned_voters'],
                              answer['t_vote_count_sec'], answer['t_quick_bounds_sec'], answer['t_pruning_sec'],
                              answer['t_solver_sec'], answer['t_total_sec']]
                    record = [str(i) for i in record]

                    out.write(sep.join(record) + '\n')

    print('Done.')


if __name__ == '__main__':
    generate_profiles()
    run_experiment()
