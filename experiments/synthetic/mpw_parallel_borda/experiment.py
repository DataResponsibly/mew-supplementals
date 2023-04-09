from multiprocessing import cpu_count

import pandas as pd

from experiments.helper import get_dir, get_rule_vector
from experiments.synthetic.posets.experiment import generate_profile_df
from ppref.preferences.poset import Poset
from ppref.profile_solvers.solver_by_voter_pruning import parallel_baseline_solver
from ppref.profile_solvers.solver_for_mpw import parallel_mpw_solver


def generate_combs():
    m_list = list(range(3, 7))
    n_list = list(range(1, 16))
    phi = 0.5
    pmax = 0.1
    batch_list = list(range(10))

    combs = set()

    # varying m
    n = 5
    for m in m_list:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    # varying n
    m = 5
    for n in n_list:
        for batch in batch_list:
            combs.add((m, n, phi, pmax, batch))

    return sorted(combs, key=lambda x: sum(x[:2]))


def generate_profiles():
    for (m, n, phi, pmax, batch) in generate_combs():
        filename = f'{m}_candidates_{n}_voters_{phi=:.1f}_pmax={pmax:.1f}_batch_{batch}.tsv'
        fullpath = get_dir(__file__) / f'profiles/{filename}'
        fullpath.parent.mkdir(parents=True, exist_ok=True)

        if not fullpath.exists():
            print(f'[INFO] Generating {filename}')
            df = generate_profile_df(m, n, phi, pmax, batch)
            df.to_csv(fullpath, index=False, sep='\t')

    print('Done.')


def run_experiment(is_improved=False):
    sep = '\t'
    t_max_min = 60 * 8

    rule_list = ['Borda']

    out_file = get_dir(__file__) / f'experiment_output_of_improved_{is_improved}.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:
        if open_mode == 'w':
            cols = ['m', 'n', 'phi', 'pmax', 'batch', 'rule', 'threads', 't_max_min', 'mpw', 't_mpw_s', 'error', 'mew', 't_mew_s']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for (m, n, phi, pmax, batch) in generate_combs():
            for rule in rule_list:

                for threads in [cpu_count(), ]:

                    condition = f'(m == {m}) and (n == {n}) and (phi == {phi}) and (pmax == {pmax}) and ' \
                                f'(batch == {batch}) and (rule == "{rule}") and (threads == {threads}) and ' \
                                f'(t_max_min == {t_max_min})'

                    is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                    if not is_existing:
                        print(f'- Executing {condition}')

                        df_in = generate_profile_df(m, n, phi, pmax, batch)

                        profile: list[Poset] = df_in['poset'].tolist()
                        ans_mpw = parallel_mpw_solver(profile, get_rule_vector(rule, m), threads=threads,
                                                      t_max_min=t_max_min, verbose=False, is_improved=is_improved)
                        ans_mew = parallel_baseline_solver(profile, get_rule_vector(rule, m))

                        if ans_mpw['has_answer']:
                            record = [m, n, f'{phi:.1f}', f'{pmax:.1f}', batch, rule, threads, t_max_min, ans_mpw['winners'],
                                      ans_mpw['t_total_sec'], 'no-error', ans_mew['winners'], ans_mew['t_sec']]
                        else:
                            record = [m, n, f'{phi:.1f}', f'{pmax:.1f}', batch, rule, threads, t_max_min, 'error', 'error',
                                      ans_mpw['error'], ans_mew['winners'], ans_mew['t_sec']]

                        record = [str(i) for i in record]

                        out.write(sep.join(record) + '\n')

    print('Done.')


def main():
    import sys
    is_improved = eval(sys.argv[1])
    run_experiment(is_improved=is_improved)


if __name__ == '__main__':
    main()
