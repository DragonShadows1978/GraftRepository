"""Prior art: RS3 (project,2026) cache-slice provenance; C2 schema boundary cases."""
from scripts.grm_c2_cells import strict_capture_grade


def node(pin='live',width=96):
    return {'capture':dict(capture_pin=pin,capture_shift=19+width if pin=='live' else None,
        capture_shift_observed=19+width,capture_shift_derived_from='arena geometry',
        n_sink=19,arena_width=width,live_shift=19+width)}


def test_default_unpinned_requires_full_registered_geometry():
    n=node('off',256)
    assert strict_capture_grade([n],'defaults',256,19)['valid']
    n['capture'].pop('capture_shift_derived_from')
    assert not strict_capture_grade([n],'defaults',256,19)['valid']
    assert not strict_capture_grade([node('off',96)],'defaults',256,19)['valid']


def test_cache_payload_not_falsely_claimed_harvest_pinned():
    n=node();n['capture'].update(capture_payload_source='live_cache_slice',
        capture_pin_moves_payload=False,capture_span_pos0=120)
    result=strict_capture_grade([n],'profile',96,19)
    assert result['valid']
    assert result['rows'][0]['grade']=='FRESH_CACHE_SLICE_PAYLOAD_NOT_MOVED_BY_PIN'
    n['capture']['capture_span_pos0']=0
    assert not strict_capture_grade([n],'profile',96,19)['valid']
