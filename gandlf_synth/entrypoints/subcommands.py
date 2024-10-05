from gandlf_synth.entrypoints.run import run as run_command
from gandlf_synth.entrypoints.construct_csv import (
    construct_csv as construct_csv_command,
)
from gandlf_synth.entrypoints.verify_install import (
    verify_install as verify_install_command,
)

cli_subcommands = {
    "run": run_command,
    "construct-csv": construct_csv_command,
    "verify-install": verify_install_command,
}
