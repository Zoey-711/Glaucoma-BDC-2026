from pathlib import Path

import pytest

from predict import submission_path


def test_round_destination():
    assert submission_path() == Path('submission/round_01/submission.csv')
    assert submission_path(round_name='round_02_B02') == Path('submission/round_02_B02/submission.csv')
    assert submission_path(output='outputs/smoke/submission.csv') == Path('outputs/smoke/submission.csv')


@pytest.mark.parametrize('name', ['../outside', 'round_01/other', 'round_1', 'C:\\outside'])
def test_round_rejects_path_escape(name):
    with pytest.raises(ValueError):
        submission_path(round_name=name)
