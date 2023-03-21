import os
import re
import pathlib
import pytest

import ase.io
import ase.calculators.emt

import ape.calculators.committee


def test_committeemember_initialize():
    ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())

    training_data = os.path.join(f'{os.path.dirname(__file__)}/data/training_data.xyz')
    ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT(),
                                              training_data=training_data)
    ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT(),
                                              training_data=ase.io.read(training_data, ':'))


def test_committeemember_add_training_data():
    member = ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())
    training_data = os.path.join(f'{os.path.dirname(__file__)}/data/training_data.xyz')

    member.add_training_data(training_data)
    member.add_training_data(pathlib.Path(training_data))
    member.add_training_data(ase.io.read(training_data, ':'))


def test_committeemember_is_sample_in_atoms():
    member = ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())
    training_data = ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/data/training_data.xyz'), ':')
    test_data = ase.io.read(os.path.join(f'{os.path.dirname(__file__)}/data/test_data.xyz'), ':')

    member.add_training_data(training_data)

    assert member.is_sample_in_atoms(sample=training_data[0])
    with pytest.raises(RuntimeError,
                       match=re.escape('Can\'t test if `sample` is in `atoms`.'
                                       '`sample` has no Atoms.info[\'_Index_FullTrainingSet\']')):
        assert not member.is_sample_in_atoms(sample=test_data[0])
    test_data[0].info['_Index_FullTrainingSet'] = -1
    assert not member.is_sample_in_atoms(sample=test_data[0])


def test_committeemember_setter():
    member = ape.calculators.committee.CommitteeMember(calculator=ase.calculators.emt.EMT())

    with pytest.raises(RuntimeError, match=re.escape('Use `add_training_data()` to modify the committee member')):
        member.filename = ''

    with pytest.raises(RuntimeError, match=re.escape('Use `add_training_data()` to modify the committee member')):
        member.atoms = []

    with pytest.raises(RuntimeError, match=re.escape('Use `add_training_data()` to modify the committee member')):
        member.ids = []
