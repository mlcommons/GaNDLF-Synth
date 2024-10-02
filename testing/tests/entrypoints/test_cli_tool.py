from click.testing import CliRunner
from gandlf_synth.entrypoints.cli_tool import gandlf_synth
from gandlf_synth.version import __version__


def test_version_command():
    runner = CliRunner()
    result = runner.invoke(gandlf_synth, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
