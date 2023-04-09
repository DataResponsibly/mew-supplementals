import pandas as pd

from experiments.helper import get_dir, get_path_to_poset_profile
from experiments.outdated.synthetic_data.profile_generator import generate_parameter_combs
from ppref.helper import get_computer_info
from ppref.preferences.poset import Poset
from ppref.profile_solvers.posets.baseline_solver import solver_baseline
from ppref.profile_solvers.posets.candidate_pruning import solver_w_candidate_pruning
from ppref.profile_solvers.posets.voter_pruning import solver_w_voter_pruning


def experiment():
    parameter_combs = generate_parameter_combs()

    out_file = get_dir(__file__) / f'experiment_output.tsv'
    open_mode = 'a' if out_file.exists() else 'w'
    with open(out_file, open_mode, buffering=1) as out:

        sep = '\t'

        if open_mode == 'w':
            out.write(f'# Calculate poset profiles, with \n'
                      f'#     1) basic voter pruning by upper lower bounds;\n'
                      f'#     2) sequential exact solver.\n')
            out.write(f'# {get_computer_info()}\n')
            cols = ['k_approval', 'num_candidates', 'num_voters', 'phi', 'rsm_pmax', 'batch', 'winners', 'winner_score',
                    't_baseline_sec', 'winners_vp', 'pruned_voters', 't_vp_sec', 'winners_cp', 'pruned_candidates',
                    't_cp_sec']
            out.write(sep.join(cols) + '\n')
        else:
            df_existing = pd.read_csv(out_file, sep=sep, comment='#')

        for num_candidates, num_voters, phi, rsm_pmax, batch in parameter_combs:
            for k_approval in [1, 2, 3, 4]:
                condition = f'(k_approval == {k_approval}) and (num_candidates == {num_candidates}) and ' \
                            f'(num_voters == {num_voters}) and (phi == {phi:.1f}) and rsm_pmax == {rsm_pmax:.1f} and ' \
                            f'(batch == {batch})'
                is_existing = (open_mode == 'a') and (not df_existing.query(condition).empty)

                if not is_existing:
                    print(condition)
                    filename = get_path_to_poset_profile(num_candidates, num_voters, phi, rsm_pmax, batch)
                    df = pd.read_csv(filename, delimiter='\t', comment='#')

                    posets: list[Poset] = [eval(po) for po in df['poset']]
                    rule_vector = [1 for _ in range(k_approval)] + [0 for _ in range(len(posets[0].item_set) - k_approval)]

                    answer_baseline = solver_baseline(posets, rule_vector=rule_vector)
                    answer_vp = solver_w_voter_pruning(posets, rule_vector=rule_vector)
                    answer_cp = solver_w_candidate_pruning(posets, rule_vector=rule_vector)

                    winners_bl = answer_baseline['winners']
                    winner_score = answer_baseline['winner_score']
                    t_baseline_sec = answer_baseline['t_sec']

                    winners_vp = answer_vp['winners']
                    num_pruned_voters = answer_vp['num_pruned_voters']
                    t_vp_sec = answer_vp['t_total_sec']

                    winners_cp = answer_cp['winners']
                    num_pruned_candidates = answer_cp['num_pruned_candidates']
                    t_cp_sec = answer_cp['t_total_sec']

                    out.write(f'{k_approval}{sep}{num_candidates}{sep}{num_voters}{sep}{phi:.1f}{sep}{rsm_pmax:.1f}{sep}'
                              f'{batch}{sep}{winners_bl}{sep}{winner_score}{sep}{t_baseline_sec}{sep}{winners_vp}{sep}'
                              f'{num_pruned_voters}{sep}{t_vp_sec}{sep}{winners_cp}{sep}{num_pruned_candidates}{sep}'
                              f'{t_cp_sec}\n')


if __name__ == '__main__':
    experiment()
