# pyseq_template
Template for PySeq Sequencers

# Manual Changes
Rename repo to the alias of the sequencer.
For example to repurpose an Illumina HiSeq 2500 an appropriate alias would be PySeq2500

pyproject.toml
    project.name = "pyseq_{sequencer alias"}
    project.authors -> add your name here
    project.version = "0.0.0-alpha"
    project.urls.Issues = "https://github.com/PySeqEcosystem/{sequencer alias}"
    project.description = "Open source control sofware for [sequencer make & model]"

.github/workflows/python-publish.yml
    jobs.pypi-publish.environment.url = https://pypi.org/project/{sequencer alias}/${{ github.event.release.name }}



