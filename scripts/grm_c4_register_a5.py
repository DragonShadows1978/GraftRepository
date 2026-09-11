#!/usr/bin/env python3
"""Create-only lead A5 registration, before CPU gates.
Prior art: local C4/A4 SHA registration (house,2026), reused without novelty.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_campaign as c
from scripts import grm_c4_remaining_a3 as a3
from scripts import grm_c4_split_a5 as a5


def main():
    previous=a3.binding()
    assert not (a5.OUT/'runs').exists(), 'Registration after execution refused'
    with a5.COMMANDS.open('x') as f:
        f.write(a5.command_text(a5.plan(previous)['scheduled_units']))
    row=a5.expected_payload(previous)
    c.write_once(a5.AMENDMENT,row)
    with a5.AMENDMENT.with_suffix('.sha256').open('x') as f:
        f.write(c.record(a5.AMENDMENT)['sha256']+'  amendment_a7.json\n')
    print(c.record(a5.AMENDMENT))


if __name__=='__main__':
    main()
