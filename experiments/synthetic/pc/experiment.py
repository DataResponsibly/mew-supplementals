from multiprocessing import Pool

import pandas as pd
from numpy.random import default_rng

from experiments.helper import get_dir, get_rule_vector
from ppref.preferences.special import PartialChain
from ppref.profile_solvers.solver_by_voter_pruning import sequential_solver_of_voter_pruning


def generate_combs():
    m = 80
    n = 1000
    batch_list = list(range(10))

    combs = set()
    for chain_size in [5, 10, 20]:
        for batch in batch_list:
            combs.add((m, n, chain_size, batch))

    chain_size = 5
    for m in [10, 20, 40, 80]:
        for batch in batch_list:
            combs.add((m, n, chain_size, batch))

    m = 80
    chain_size = 5
    for n in [100, 1_000, 10_000, 100_000]:
        for batch in batch_list:
            combs.add((m, n, chain_size, batch))

    return sorted(combs, key=lambda x: x[1])


def generate_profile_df(m, n, chain_size, batch):
    item_set = set(range(m))
    rng = default_rng([m, n, chain_size, batch])
    df = pd.DataFrame(columns=['m', 'chain_size', 'ppmw'])
    for i in range(n):
        pc = PartialChain.generate_a_random_instance(chain_size, item_set, rng)
        df.loc[i] = [m, chain_size, repr(pc)]

    return df


def worker(m, n, chain_size, batch):
    filename = f'{m}_candidates_{n}_voters_chain_size_{chain_size}_batch_{batch}.tsv'
    fullpath = get_dir(__file__) / f'profiles/{filename}'
    fullpath.parent.mkdir(parents=True, exist_ok=True)

    if not fullpath.exists():
        print(f'[INFO] Generating {filename}')
        df = generate_profile_df(m, n, chain_size, batch)
        df.to_csv(fullpath, index=False, sep='\t')


def generate_profiles(verbose=False):
    with Pool(processes=10) as pool:
        pool.starmap(worker, generate_combs())

    print('[INFO] Profile generation... Done.')


def run_experiment():
    sep = '\t'

    threads = 1
    grouping = True

    rule_list = ['Plurality', '2-approval', 'Borda']

    out_file = get_dir(__file__) / 'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'chain_size', 'batch', 'rule', 'threads', 'grouping', 'winners', 'score_upper', 'score_lower',
                    'num_pruned_voters', 't_quick_bounds_sec', 't_pruning_sec', 't_solver_sec', 't_total_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, chain_size, batch) in generate_combs():
            for rule in rule_list:

                condition = f'(m == {m}) and (n == {n}) and (chain_size == {chain_size}) and (batch == {batch}) ' \
                            f'and (rule == "{rule}") and (threads == {threads}) and (grouping == {grouping})'

                is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                if not is_existing:
                    print(f'- Executing {condition}')

                    filename = f'{m}_candidates_{n}_voters_chain_size_{chain_size}_batch_{batch}.tsv'
                    fullpath = get_dir(__file__) / f'profiles/{filename}'
                    df_in = pd.read_csv(fullpath, sep=sep)

                    profile: list[PartialChain] = [eval(ppmw) for ppmw in df_in['ppmw']]
                    answer = sequential_solver_of_voter_pruning(profile, get_rule_vector(rule, m))

                    record = [m, n, chain_size, batch, rule, threads, grouping, answer['winners'], answer['score_upper'],
                              answer['score_lower'], answer['num_pruned_voters'], answer['t_quick_bounds_sec'],
                              answer['t_pruning_sec'], answer['t_solver_sec'], answer['t_total_sec']]
                    record = [str(i) for i in record]

                    out.write(sep.join(record) + '\n')

    print('Done.')


if __name__ == '__main__':
    generate_profiles()
    run_experiment()
