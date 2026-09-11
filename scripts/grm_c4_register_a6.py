#!/usr/bin/env python3
"""Create-only A6 registration, before gates.
Prior art: local C4/A5 SHA registration (house, 2026), reused without novelty.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_skip_a6 as a6
from scripts import grm_c4_split_a5 as a5


def main():
    previous = a6.LEGACY_BINDING()
    assert not (a5.OUT/'runs').exists(), 'Registration after A5 execution refused'
    assert not (a6.OUT/'runs').exists(), 'Registration after A6 execution refused'
    assert all(c.record(p['path']) == p for p in c.read(a6.OUT/'before.json')['files']), 'Pre-registration drift'
    with a6.COMMANDS.open('x') as f:
        f.write(a6.command_text(a6.plan(previous)['scheduled_units']))
    c.write_once(a6.AMENDMENT, a6.expected_payload(previous))
    with a6.AMENDMENT.with_suffix('.sha256').open('x') as f:
        f.write(c.record(a6.AMENDMENT)['sha256']+'  amendment_a8.json\n')
    print(c.record(a6.AMENDMENT))


if __name__ == '__main__':
    main()
