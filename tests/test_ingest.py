import zipfile

import pytest

from scripts.prepare_official import safe_member, manifest


@pytest.mark.parametrize('name', ['../escape.jpg', '/absolute.jpg', 'C:/escape.jpg', 'a\\..\\escape.jpg'])
def test_zip_escape_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        safe_member(name, tmp_path)


def test_schema_mapping(tmp_path):
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('glaucoma_first/train.csv', 'Image_name,Age,True_Label\n001.jpg,53.0,1\n')
        z.writestr('glaucoma_first/train/001.JPG', b'fixture only')
    with zipfile.ZipFile(archive) as z:
        frame, columns = manifest(z, 'train')
    assert frame.iloc[0].to_dict() == {'image_id': '001.jpg', 'path': 'glaucoma_first/train/001.JPG', 'label': '1'}
    assert 'patient_id' not in frame


def test_missing_archive_image(tmp_path):
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('glaucoma_first/valid.csv', 'Image_name,True_Label\na.jpg,0\n')
    with zipfile.ZipFile(archive) as z, pytest.raises(ValueError, match='missing'):
        manifest(z, 'valid')


def test_cost_case_mapping():
    from pathlib import Path
    from cost import benchmark_ids
    assert benchmark_ids([Path('test_00000.JPG')], ['test_00000.jpg']) == ['test_00000.jpg']
    with pytest.raises(ValueError, match='Ambiguous'):
        benchmark_ids([], ['a.jpg', 'a.JPG'])
