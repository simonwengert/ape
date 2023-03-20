import numpy as np
import ase.io


def subsample(inputs, output_files, M, N_s, keep_isolated_atoms=True):
    """
    Draw subsamples (without replacement) from given configurations.

    Parameter:
    ----------
    inputs: list(Atoms)
        List of configurations to draw subsamples from
        (e.g. full training set).
    output_files: list(str / Path)
        Target locations for subsampled sets of configurations.
    M: int
        Number of subsets to be drawn.
    N_s: int
        Number of samples per subsets.
    keep_isolated_atoms: bool, default True
        Make isolated atoms (if present) be part of each subset.

    Returns:
    --------
    list(list(Atoms)) with subsampled sets of configurations.

    """
    # keep track of position in original set of configurations
    samples = []
    isolated_atoms = []  # keep isolated atoms for each subsample
    for idx_i, atoms_i in enumerate(inputs):
        atoms_i.info['_Index_FullTrainingSet'] = idx_i
        if keep_isolated_atoms and len(atoms_i) == 1:
            isolated_atoms.append(atoms_i)
        else:
            samples.append(atoms_i)

    N_s -= len(isolated_atoms)
    assert 1 < N_s <= len(samples), 'Negative N_s (after reduction by number of isolated atoms)'
    assert len(output_files) == M, f'`outputs` requires `M` files to be specified.'

    indice_pool = np.arange(len(samples))
    subsample_indices = []
    for _ in range(M):
        if N_s <= len(indice_pool):
            selected_indices = np.random.choice(indice_pool, N_s, False)
            indice_pool = indice_pool[~np.isin(indice_pool, selected_indices)]
            subsample_indices.append(selected_indices)
        else:
            selected_indices_part_1 = indice_pool
            # re-fill pool with indices, taking account of already selected ones,
            # in order to avoid dublicate selections
            indice_pool = np.arange(len(samples))
            indice_pool = indice_pool[~np.isin(indice_pool, selected_indices_part_1)]
            selected_indices_part_2 = np.random.choice(indice_pool, N_s-len(selected_indices_part_1), False)
            indice_pool = indice_pool[~np.isin(indice_pool, selected_indices_part_2)]
            selected_indices = np.concatenate((selected_indices_part_1, selected_indices_part_2))
            subsample_indices.append(selected_indices)

    subsamples = [isolated_atoms + [samples[idx_i] for idx_i in idxs] for idxs in subsample_indices]
    for output_file_i, subsample_i in zip(output_files, subsamples):
        ase.io.write(output_file_i, subsample_i)

    return subsamples
