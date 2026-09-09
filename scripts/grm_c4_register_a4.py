"""Create-only cap registration before gates.
Prior art: C4/A5 registration (house, 2026), reused without novel algorithm.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import grm_c4_cap_a4 as cap
from scripts import grm_c4_campaign as c


def main():
    c.binding()
    row = cap.expected_payload()
    c.write_once(cap.AMENDMENT, row)
    pin = c.record(cap.AMENDMENT)
    with cap.AMENDMENT.with_suffix('.sha256').open('x') as f:
        f.write(pin['sha256'] + '  amendment_a6.json\n')
    print(pin)


if __name__ == '__main__':
    main()
