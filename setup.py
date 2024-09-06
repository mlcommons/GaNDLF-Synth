import re
import os
import sys
from setuptools import setup, find_packages


try:
    with open("README.md") as readme_file:
        readme = readme_file.read()
except Exception as error:
    readme = "No README information found."
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % ("README.md", error))


try:
    filepath = "gandlf_synth/version.py"
    version_file = open(filepath)
    (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())

except Exception as error:
    __version__ = "0.0.1"
    sys.stderr.write("Warning: Could not open '%s' due %s\n" % (filepath, error))

dockerfiles = [
    item
    for item in os.listdir(os.path.dirname(os.path.abspath(__file__)))
    if (os.path.isfile(item) and item.startswith("Dockerfile-"))
]

extra_files = []
toplevel_package_excludes = ["testing*"]

black_version = "23.11.0"
requirements = [
    f"black=={black_version}",
    "GANDLF@git+https://github.com/mlcommons/GandLF.git@master",
    "lightning==2.4.0",
    "monai-generative==0.2.3",
]
if __name__ == "__main__":
    setup(
        name="gandlf_synth",
        version=__version__,
        author="MLCommons",
        author_email="gandlf@mlcommons.org",
        python_requires=">=3.8, <3.12",
        packages=find_packages(where=os.path.dirname(os.path.abspath(__file__))),
        entry_points={
            "console_scripts": [
                "gandlf-synth = gandlf_synth.entrypoints.cli_tool:gandlf_synth"
            ]
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
        ],
        description=(
            "PyTorch-based framework that handles image synthesis using various DL architectures for medical imaging."
        ),
        install_requires=requirements,
        license="Apache-2.0",
        long_description=readme,
        long_description_content_type="text/markdown",
        include_package_data=True,
        package_data={"gandlf_synth": extra_files},
        keywords="image generation, image synthesis, data-augmentation, medical-imaging, clinical-workflows, deep-learning, pytorch",
        zip_safe=False,
    )
