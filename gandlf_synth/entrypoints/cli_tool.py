import click
from gandlf_synth.entrypoints.subcommands import cli_subcommands
from gandlf_synth.entrypoints import append_copyright_to_help
from gandlf_synth import version


@click.group()
@click.version_option(
    version, "--version", "-v", message="ganldf_synth version: %(version)s"
)
@click.pass_context  # Pass the context to subcommands
@append_copyright_to_help
def gandlf_synth(ctx):
    """gandlf-synth command-line tool."""
    ctx.ensure_object(dict)


# registers subcommands: `gandlf anonymizer`, `gandlf-synth run`, etc.
for command_name, command in cli_subcommands.items():
    gandlf_synth.add_command(command, command_name)

if __name__ == "__main__":
    # pylint: disable=E1120
    gandlf_synth()
