from time import time

import pandas as pd
from numpy.random import default_rng

from experiments.helper import get_dir, get_random_poset_by_rsm
from ppref.models.mallows import Mallows
from ppref.preferences.poset import Poset
from ppref.rank_estimators.rim_w_poset import calculate_rank_probs_for_item_given_rim_w_poset_by_parallel


def generate_profile():
    rng = default_rng(0)

    num_candidates_list = [10, 15, 20]
    num_candidate_to_mallows = {m: Mallows(reference=tuple(range(m)), phi=1.0) for m in num_candidates_list}
    rsm_pmax_list = [0.1, 0.5, 0.9]
    repeat = 5

    df = pd.DataFrame(columns=['m', 'rsm_pmax', 'ith_poset', 'cardinality', 'poset'])

    for rep in range(repeat):
        for m in num_candidates_list:
            for rsm_pmax in rsm_pmax_list:
                mallows = num_candidate_to_mallows[m]
                ranking = mallows.sample_a_ranking(rng)
                probs = rng.random(size=m) * rsm_pmax
                poset = get_random_poset_by_rsm(ranking, probs, rng)
                cardinality = poset.dag.number_of_nodes()

                df.loc[df.shape[0]] = [m, rsm_pmax, rep, cardinality, poset]

    df.to_csv(get_dir(__file__) / 'experiment_input.tsv', index=False, sep='\t')


def run_experiment():
    sep = '\t'
    threads = 1

    df_posets = pd.read_csv(get_dir(__file__) / 'experiment_input.tsv', sep='\t')
    df_posets.sort_values(by=['ith_poset', 'm'], inplace=True)

    out_file = get_dir(__file__) / 'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:

        if open_mode == 'w':
            cols = ['m', 'rsm_pmax', 'k_approval', 'ith_poset', 'cardinality', 'cover_width', 'num_states', 'item',
                    'threads', 'time_s', 'error']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep)

        for _, row in df_posets.iterrows():
            m = row['m']
            p_max = row['rsm_pmax']
            ith = row['ith_poset']
            poset: Poset = eval(row['poset'])
            item = next(poset.get_generator_of_linears()).r[0]

            for k in [1, 2, 5, 10]:

                condition = f'(m == {m}) and (rsm_pmax == {p_max}) and (k_approval == {k}) and (ith_poset == {ith}) and ' \
                            f'(item == {item}) and (threads == {threads})'
                is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                if not is_existing:
                    print(f'- Executing ith_poset={ith}, {m=}, {p_max=}, {item=}, {k=}, {threads=}')

                    cardinality = row['cardinality']
                    mallows = Mallows(tuple(range(m)), 1.0)

                    t1 = time()
                    ans = calculate_rank_probs_for_item_given_rim_w_poset_by_parallel(item=item, rim=mallows,
                                                                                      poset=poset, threads=threads,
                                                                                      t_max_min=600, max_rank=k - 1)
                    time_s = time() - t1
                    ps = list(ans.get('probabilities', []))
                    cover_width = ans.get('cover_width', -1)
                    num_states = ans.get('num_states', -1)
                    error = ans.get('error', 'none')

                    out.write(f'{m}{sep}{p_max}{sep}{k}{sep}{ith}{sep}{cardinality}{sep}{cover_width}{sep}{num_states}{sep}'
                              f'{item}{sep}{threads}{sep}{time_s}{sep}{error}\n')


if __name__ == '__main__':
    # generate_profile()
    run_experiment()
