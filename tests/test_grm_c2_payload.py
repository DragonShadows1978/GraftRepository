"""Prior art: B3 ordinary manifest durability, project (2026); C2 input guards."""
import pytest
from scripts.grm_c2_profile import create
from scripts.grm_c2_cells import assert_payloads


def test_missing_pending_payload_cannot_trigger_reharvest(tmp_path):
    create(tmp_path/'manifest.json',{'nodes':[{'retired':False,'payload_pending':True}]})
    with pytest.raises(ValueError,match='reharvest forbidden'):assert_payloads(tmp_path)


def test_missing_durable_file_is_red(tmp_path):
    create(tmp_path/'manifest.json',{'nodes':[{'retired':False,'payload_pending':False}]})
    with pytest.raises(ValueError,match='reharvest forbidden'):assert_payloads(tmp_path)
    (tmp_path/'nodes').mkdir();(tmp_path/'nodes/0000.npz').write_bytes(b'payload sentinel')
    assert len(assert_payloads(tmp_path))==1
