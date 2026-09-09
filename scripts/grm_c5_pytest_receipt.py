"""Prior art: GRM WC1 CPU receipts (2026); ours fingerprints pytest's actual imports."""
import os
from pathlib import Path

from scripts.grm_c5_offline import receipt, registration, write_new


def pytest_sessionfinish(session, exitstatus):
    registration()
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = {key: len(value) for key, value in reporter.stats.items()}
    result = receipt("cpu-gate", {"exitstatus": int(exitstatus),
                                  "tests_collected": session.testscollected,
                                  "stats": stats,
                                  "pytest_args": session.config.invocation_params.args})
    write_new(Path(os.environ["GRM_C5_PYTEST_RECEIPT"]), result)
