#!usr/bin/env python
# -*- coding: utf-8 -*-

import click

from gandlf_synth.entrypoints import append_copyright_to_help


@click.command()
@append_copyright_to_help
def verify_install():
    try:
        import gandlf_synth as gfs

        print("GaNDLF-Synth installed version:", gfs.__version__)
    except Exception as e:
        raise Exception(
            "GaNDLF-Synth not properly installed, please see https://github.com/mlcommons/GaNDLF-Synth/tree/main/docs/setup"
        ) from e

    print(
        "GaNDLF-Synth is ready. See https://github.com/mlcommons/GaNDLF-Synth/tree/main/docs/usage"
    )


if __name__ == "__main__":
    verify_install()
