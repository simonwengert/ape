import os
import re
import copy
import pathlib
import pytest
import numpy as np

import ase.io
import ase.calculators.emt
import quippy.potential

import ape.calculators.committee


@pytest.fixture
def committeemember():
    member = ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())
    training_data = os.path.join(f'{os.path.dirname(__file__)}/data/training_data_minimal.xyz')
    member.set_training_data(training_data)
    return member


@pytest.fixture
def committee_minimal(committeemember):
    committee = ape.calculators.committee.Committee(
        members=[copy.deepcopy(committeemember), copy.deepcopy(committeemember)]
    )
    return committee


@pytest.fixture
def committee():
    committee = ape.calculators.committee.Committee()
    for idx_i in range(10):
        basepath_i = os.path.join(f'{os.path.dirname(__file__)}/data/committee_{idx_i}')
        committee += ape.calculators.committee.CommitteeMember(
            calculator=quippy.potential.Potential(param_filename=f'{basepath_i}/fit/GAP.xml'),
            training_data=f'{basepath_i}/train.xyz'
        )
    return committee


def test_committeemember_initialize():
    ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())

    training_data = os.path.join(f'{os.path.dirname(__file__)}/data/training_data_minimal.xyz')
    ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT(),
                                              training_data=training_data)
    ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT(),
                                              training_data=ase.io.read(training_data, ':'))


def test_committeemember_set_training_data(committeemember):
    training_data = os.path.join(f'{os.path.dirname(__file__)}/data/training_data_minimal.xyz')

    with pytest.warns(Warning, match=re.escape('Overwriting current training data.')):
        committeemember.set_training_data(training_data)
    with pytest.warns(Warning, match=re.escape('Overwriting current training data.')):
        committeemember.set_training_data(pathlib.Path(training_data))
    with pytest.warns(Warning, match=re.escape('Overwriting current training data.')):
        committeemember.set_training_data(ase.io.read(training_data, ':'))


def test_committeemember_is_sample_in_atoms(committeemember):
    training_data = ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/data/training_data_minimal.xyz'), ':')
    test_data = ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/data/test_data.xyz'), ':')

    assert committeemember.is_sample_in_atoms(sample=training_data[0])
    with pytest.raises(RuntimeError,
                       match=re.escape('Can\'t test if `sample` is in `atoms`. '
                                       '`sample` has no Atoms.info[\'_Index_FullTrainingSet\']')):
        assert not committeemember.is_sample_in_atoms(sample=test_data[0])
    test_data[0].info['_Index_FullTrainingSet'] = -1
    assert not committeemember.is_sample_in_atoms(sample=test_data[0])


def test_committeemember_setter(committeemember):

    with pytest.raises(RuntimeError, match=re.escape('Use `set_training_data()` to modify the committee member')):
        committeemember.filename = ''

    with pytest.raises(RuntimeError, match=re.escape('Use `set_training_data()` to modify the committee member')):
        committeemember.atoms = []

    with pytest.raises(RuntimeError, match=re.escape('Use `set_training_data()` to modify the committee member')):
        committeemember.ids = []


def test_committee_initialize(committeemember):
    committee = ape.calculators.committee.Committee()
    expected_status = [
        ('members', []),
        ('number', 0),
        ('atoms', []),
        ('ids', []),
        ('alphas', {}),
        ('calibrated_for', set()),
    ]
    for attribute_i, value_i in expected_status:
        assert getattr(committee, attribute_i) == value_i
    with pytest.warns(Warning, match=re.escape('`Committee.set_internal_validation_set()` has not been called or '
                                               '`Committee`-instance has been altered since last call.')):
        assert getattr(committee, 'validation_set') == []

    member_0 = copy.deepcopy(committeemember)
    member_1 = copy.deepcopy(committeemember)
    committee = ape.calculators.committee.Committee(
        members=[member_0, member_1]
    )
    expected_status = [
        ('members', [member_0, member_1]),
        ('number', 2),
        ('atoms', member_0.atoms + member_1.atoms),
        ('ids', member_0.ids + member_1.ids),
        ('alphas', {}),
        ('calibrated_for', set()),
    ]
    for attribute_i, value_i in expected_status:
        assert getattr(committee, attribute_i) == value_i
    with pytest.warns(Warning, match=re.escape('`Committee.set_internal_validation_set()` has not been called or '
                                               '`Committee`-instance has been altered since last call.')):
        assert getattr(committee, 'validation_set') == []


def test_committee_member(committee_minimal):

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        ape.calculators.committee.Committee(members=[0, 1])

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        committee_minimal.members = [0, 1]

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        committee_minimal.add_member(0)

    with pytest.raises(AssertionError,
                       match=re.escape('Members of `Committee` need to be of type `CommitteeMember`. Found ')):
        committee_minimal += 0


def test_committee_set_internal_validation_set(committee):

    with pytest.raises(AssertionError):
        committee.set_internal_validation_set(0)

    with pytest.raises(AssertionError):
        committee.set_internal_validation_set(committee.number - 1)

    committee.set_internal_validation_set(appearance_threshold=3)
    obtained = set([atoms_i.info['_Index_FullTrainingSet'] for atoms_i
                    in committee.validation_set])
    expected = set([atoms_i.info['_Index_FullTrainingSet'] for atoms_i
                    in ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/data/validation_set.xyz'), ':')])
    assert obtained == expected


def test_committee_calibrate(committee):
    committee.set_internal_validation_set(appearance_threshold=3)

    committee.calibrate(prop='energy', key='E_dftbplus_d4', location='info')
    assert committee.calibrated_for == set(['energy'])
    np.testing.assert_array_almost_equal(committee.alphas['energy'], 0.8717093014393091, decimal=6)

    committee.calibrate(prop='forces', key='F_dftbplus_d4', location='arrays')
    assert committee.calibrated_for == set(['energy', 'forces'])
    np.testing.assert_array_almost_equal(committee.alphas['forces'], 8.960552103437163, decimal=6)

    with pytest.warns(Warning,
                      match=re.escape('`alphas` will be reset to avoid inconsistencies with new validation set.')):
        committee.set_internal_validation_set(appearance_threshold=4)
        assert committee.alphas == {}


def test_committee__calculate_alpha(committee):
    vals_ref = np.array([1.01, 1.02, 1.03])
    vals_pred = np.array([2.01, 1.02, 1.03])
    var_pred = np.array([1.01, 0.02, 0.03])

    obtained = committee._calculate_alpha(vals_ref, vals_pred, var_pred)
    np.testing.assert_array_almost_equal(obtained, 0.39584382766472004, decimal=6)


def test_committee_scale_uncertainty(committee):
    committee._alphas = {'energy': 2.5}

    assert committee.scale_uncertainty(2, 'energy') == 5.0
