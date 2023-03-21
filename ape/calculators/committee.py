"""
Committee of Models

Calculate properties with a list of models and saves them into info/arrays.
"""
import warnings
import numpy as np
from pathlib import Path
from collections import Counter

import ase.io
from ase.calculators.calculator import Calculator, all_changes


__default_properties = ['energy', 'forces', 'stress']


class CommitteeUncertainty(Calculator):
    """
    Calculator for a committee of machine learned interatomic potentials (MLIP).

    The class assumes individual members of the committee already exist (i.e. their
    training is performed externally). Instances of this class are initialized with
    these committee members and results (energy, forces) are calculated as average
    over these members. In addition to these values, also the uncertainty (standard
    deviation) is calculated.

    The idea for this Calculator class is based on the following publication:
    Musil et al., J. Chem. Theory Comput. 15, 906−915 (2019)
    https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b00959

    Parameter:
    ----------
    committee: Committee-instance
        Representation for a collection of Calculators.
    atoms : ase-Atoms
        Optional object to which the calculator will be attached.
    """

    def __init__(self, committee, atoms=None):

        self.implemented_properties = ['energy', 'forces', 'stress']

        self.committee = committee

        super().__init__(atoms=atoms)

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
        """Calculates committee (mean) values and variances."""

        super().calculate(atoms, properties, system_changes)

        property_committee = {k_i: [] for k_i in properties}

        for cm_i in self.committee.members:
            cm_i.calculator.calculate(atoms=atoms, properties=properties, system_changes=system_changes)

            for p_i in properties:
                property_committee[p_i].append(cm_i.calculator.results[p_i])

        for p_i in properties:
            self.results[p_i] = np.mean(property_committee[p_i], axis=0)
            self.results[f'{p_i}_uncertainty'] = np.sqrt(np.var(property_committee[p_i], ddof=1, axis=0))

            if self.committee.is_calibrated_for(p_i):
                self.results[f'{p_i}_uncertainty'] = self.committee.scale_uncertainty(self.results[f'{p_i}_uncertainty'], p_i)
            else:
                warnings.warn(f'Uncertainty estimation has not been calibrated for {p_i}.')


class Committee:
    """
    Instances of this class represent a committee of models.

    It's use is to store the ```CommitteeMembers``` representing the committee model
    and to calibrate the obtained uncertainties (required when sub-sampling is used
    to create the training data of the committee members).

    Parameter:
    ----------
    members: list(M)
        List of ```CommitteeMember``` instances representing the committee model.
    """
    def __init__(self, members=[]):
        self.members = members
        self._update()

    @property
    def number(self):
        """Number of committee members."""
        return self._number

    @property
    def atoms(self):
        """Combined Atoms/samples in the committee."""
        return self._atoms

    @property
    def ids(self):
        """Identifiers of atoms/samples in the committee."""
        return self._ids

    @property
    def id_to_atoms(self):
        """Dictionary to translate identifiers to Atoms-objects."""
        return self._id_to_atoms

    @property
    def id_counter(self):
        """Counter-object for identifier appearances in the committee."""
        return self._id_counter

    @property
    def alphas(self):
        """(Linear) scaling factors for committee uncertainties."""
        return self._alphas

    @property
    def calibrated_for(self):
        """Set of properties the committee has been calibrated for."""
        return self._calibrated_for

    @property
    def validation_set(self):
        """List of Atoms-objects."""
        if self._validation_set:
            return self._validation_set
        else:
            raise AttributeError('`Committee`-instance has been altered since last call '
                                 'of `Committee.set_internal_validation_set()`.')

    def _update(self):
        self._number = len(self.members)
        self._atoms = [atoms_ij for cm_i in self.members for atoms_ij in cm_i.atoms]
        self._ids = [id_ij for cm_i in self.members for id_ij in cm_i.ids]
        self._id_to_atoms = {id_i: atoms_i for id_i, atoms_i in zip(self.ids, self.atoms)}
        self._id_counter = Counter(self.ids)
        self._validation_set = []
        self._alphas = {}
        self._calibrated_for = set()

    def add_member(self, member):
        """Extend committee by new ```member``` (i.e. CommitteeMember-instance)."""
        self.members.append(member)
        self._update()

    def __add__(self, member):
        """Extend committee by new ```member``` (i.e. CommitteeMember-instance)."""
        self.add_member(member)
        return self

    def __repr__(self):
        s = ''

        s_i = f'Committee Status\n'
        s += s_i
        s += '='*len(s_i) + '\n\n'

        s += f'# members:                    {self.number:>10d}\n'
        s += f'# atoms:                      {len(self.atoms):>10d}\n'
        s += f'# ids:                        {len(self.ids):>10d}\n'
        s += f'# atoms validation set:       {len(self._validation_set):>10d}\n'
        s += f'calibrated for:\n'
        for p_i in sorted(self.calibrated_for):
            s += f'{"":>4s}{p_i:<18}{self.alphas[p_i]:>18}\n'

        for idx_i, cm_i in enumerate(self.members):
            s += '\n\n'
            s_i = f'Committee Member {idx_i}:\n'
            s += s_i
            s += '-'*len(s_i) + '\n'
            s += cm_i.__repr__()

        return s

    def set_internal_validation_set(self, appearance_threshold):
        """
        Define a validation set based on the Atoms-objects of sub-sampled committee training sets.

        appearance_threshold: int
            Number of times a sample for the validation set
            is maximally allowed to appear in the training set
            of a committee member.
        """

        assert appearance_threshold <= self.number - 2

        self._validation_set = []
        for id_i, appearance_i in self.id_counter.most_common()[::-1]:
            if appearance_i <= appearance_threshold:
                break
            self._validation_set.append(self.id_to_atoms[id_i])

    def calibrate(self, prop, key, location, system_changes=all_changes):
        """
        Obtain parameters to properly scale committee uncertainties and make
        them available as an attribute (```alphas```) with another associated
        attribute (```calibrated_for```) providing information about the property
        for which the uncertainty will be scaled by it.

        properties: list(str)
            Properties for which the calibration will determine scaling factors.
        key: str
            Key under which the reference values in the validation set are stored
            (i.e. under Atoms.info[```key```] / Atoms.arrays[```key```]).
        location: str
            Either 'info' or 'arrays'.
        """
        assert location in ['info', 'arrays'], f'`location` must be \'info\' or \'arrays\', not \'{location}\'.'

        validation_ref = [np.asarray(getattr(sample_i, location)[key]).flatten() for sample_i in self.validation_set]
        validation_pred, validation_pred_var = [], []

        for idx_i, sample_i in enumerate(self.validation_set):

            sample_committee_pred = []

            for cm_j in self.members:

                if cm_j.is_sample_in_atoms(sample_i):
                    continue

                cm_j.calculator.calculate(atoms=sample_i, properties=[prop], system_changes=system_changes)
                sample_committee_pred.append(cm_j.calculator.results[prop])

            validation_pred.append(np.mean(sample_committee_pred, axis=0).flatten())
            validation_pred_var.append(np.var(sample_committee_pred, ddof=1, axis=0).flatten())

        # For symmetry-reasons it can happen that e.g. all values for a force component of an atom are equal.
        # This would lead to a division-by-zero error in self._get_alpha() due to zero-variances.
        validation_ref = np.concatenate(validation_ref)
        validation_pred = np.concatenate(validation_pred)
        validation_pred_var = np.concatenate(validation_pred_var)
        ignore_indices = np.where(validation_pred_var==0)[0]
        validation_ref = np.delete(validation_ref, ignore_indices)
        validation_pred = np.delete(validation_pred, ignore_indices)
        validation_pred_var = np.delete(validation_pred_var, ignore_indices)

        self._alphas.update(
                {prop: self._get_alpha(vals_ref=validation_ref,
                                       vals_pred=validation_pred,
                                       var_pred=validation_pred_var,
                                       )
                })
        self._calibrated_for.add(prop)

    def is_calibrated_for(self, prop):
        """Check whether committee has been calibrated for ```prop```."""
        return prop in self._calibrated_for

    def _get_alpha(self, vals_ref, vals_pred, var_pred):
        """
        Get (linear) uncertainty scaling factor alpha.

        This implementation is based on:
        Imbalzano et al., J. Chem. Phys. 154, 074102 (2021)
        https://doi.org/10.1063/5.0036522

        Parameter:
        ----------
        vals_ref: ndarray(N)
            Reference values for validation set samples.
        vals_pred: ndarray(N)
            Values predicted by the committee for validation set samples.
        var_pred: ndarray(N)
            Variance predicted by the committee for validation set samples.

        Returns:
        --------
        (Linear) uncertainty scaling factor alpha.
        """
        N_val = len(vals_ref)
        M = self.number
        alpha_squared = -1/M + (M - 3)/(M - 1) * 1/N_val * np.sum(np.power(vals_ref-vals_pred, 2) / var_pred)
        assert alpha_squared > 0, f'Obtained negative value for `alpha_squared`: {alpha_squared}'
        return np.sqrt(alpha_squared)

    def scale_uncertainty(self, value, prop):
        """
        Scale uncertainty ```value``` obtained with the committee according to the calibration
        for the corresponding property (```prop```).

        Parameter:
        ----------
        value: float / ndarray
            Represents the uncertainty values (e.g. energy, forces) to be scaled.
        prop: str
            The property associated with ```value``` (for which the committee needs to be calibrated).

        Returns:
        --------
        Scaled input ```value```.
        """
        return self.alphas[prop] * value


class CommitteeMember:
    """
    Lightweight class defining a member (i.e. a sub-model) of a committee model.

    Parameter:
    ----------
    calculator: Calculator
        Instance of a Calculator-class (or heirs e.g. quippy.potential.Potential)
        representing a machine-learned model.
    training_data: str / Path / list(Atoms), optional default=None
        Path to or Atoms of (sub-sampled) training set used to create the machine-learned model
        defined by the ```calculator```.
    """
    def __init__(self, calculator, training_data=None):
        self._calculator = calculator

        if training_data is not None:
            self.add_training_data(training_data)

    @property
    def calculator(self):
        """Model of the committee member."""
        return self._calculator

    @property
    def filename(self):
        """Path to the atoms/samples in the committee member."""
        return self._filename

    @property
    def atoms(self):
        """Atoms/samples in the committee member."""
        return self._atoms

    @property
    def ids(self):
        """Identifiers of atoms/samples in the committee member."""
        return self._ids

    def add_training_data(self, training_data):
        """
        Read in and store the training data of this committee members from the passed ```filename```.

        Parameter:
        ----------
        training_data: str / Path / list(Atoms), optional default=None
            Path to or Atoms of (sub-sampled) training set used to create the machine-learned model
            defined by the ```calculator```.
        """
        if isinstance(training_data, (str, Path)):
            self._filename = Path(training_data)
            self._atoms = ase.io.read(self.filename, ':')
        elif isinstance(training_data, list):
            self._filename = 'No Filename'
            self._atoms = training_data
        self._ids = [atoms_i.info['_Index_FullTrainingSet'] for atoms_i in self.atoms]

    def is_sample_in_atoms(self, sample):
        """Check if passed Atoms-object is part of this committee member (by comparing identifiers)."""
        return sample.info['_Index_FullTrainingSet'] in self.ids

    def __repr__(self):
        s = ''
        s += f'calculator: {str(self.calculator.__class__):>60s}\n'
        s += f'filename:   {str(self.filename):>60s}\n'
        s += f'# Atoms:    {len(self.atoms):>60d}\n'
        s += f'# IDs:      {len(self.ids):>60d}'
        return s

