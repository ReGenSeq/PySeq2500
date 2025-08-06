---
layout: default
---

# Contributing to Open Source Guides

Thanks for checking out the PySeq Ecosystem. We're excited to hear and learn from you. Your experiences will benefit others who read and use these guides.

We've put together the following guidelines to help you figure out where you can best be helpful.

## Table of Contents

0. [Types of contributions we're looking for](#types-of-contributions-were-looking-for)
0. [Ground rules & expectations](#ground-rules--expectations)
0. [How to contribute](#how-to-contribute)
0. [Style guide](#style-guide)
0. [Setting up your environment](#setting-up-your-environment)
0. [Community](#community)

## Types of contributions we're looking for

There are many ways you can directly contribute to the PySeq Ecosystem:

* Repurpose other DNA sequencers (HiSeq X, NovaSeq 6000, or any other system)
* Automate experiments on a sequencer
* Enhance / edit documentation


## Ground rules & expectations

Before we get started, here are a few things we expect from you (and that you should expect from others):

* Be kind and thoughtful in your conversations around this project. We all come from different backgrounds and projects, which means we likely have different perspectives on "how open source is done." Try to listen to others rather than convince them that your way is correct.
* The PySeq Ecosystem follows a [Contributor Code of Conduct](./CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.
* Please ensure that your contribution passes all tests if you open a pull request. If there are test failures, you will need to address them before we can merge your contribution.

## How to contribute

If you'd like to contribute, start by searching through the [pull requests](https://github.com/krpandit/pyseq_core/pulls) and [issues](https://github.com/krpandit/pyseq_core/issues) to see whether someone else has raised a similar idea or question.

If you don't see your idea listed, and you think it fits into the goals of PySeq Ecosystem, open a pull request.

A pull request doesn’t have to represent finished work. It’s usually better to open a pull request early on, so others can watch or give feedback on your progress. Just open it as a “draft” or mark as a “WIP” (Work in Progress) in the subject line or “Notes to Reviewers” sections if provided (or you can just create your own. Like this: **## Notes to Reviewer**). You can always add more commits later.

Here’s how to submit a pull request:

* [Fork the repository](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) and clone it locally. Connect your local to the original “upstream” repository by adding it as a remote. Pull in changes from “upstream” often so that you stay up to date so that when you submit your pull request, merge conflicts will be less likely. (See more detailed instructions [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).)
* [Create a branch](https://docs.github.com/en/get-started/using-github/github-flow) for your edits.
* **Reference any relevant issues** or supporting documentation in your PR (for example, “Closes #37.”)
* **Test your changes!** Run your changes against any existing tests if they exist and create new ones when needed. It’s important to make sure your changes don’t break the existing project.


## Setting up your environment

The PySeq ecosystems uses [uv](https://docs.astral.sh/uv/) for package and project management. 

Once you have uv setup, forked a respository, and cloned it locally, run:
```bash
uv sync
uv run pre-commit install
```

## Community

Discussions about the PySeq Ecosystem take place on this repository's [Discussions](https://github.com/krpandit/pyseq_core/discussions) section. Anybody is welcome to join these conversations.

Wherever possible, do not take these conversations to private channels, including contacting the maintainers directly. Keeping communication public means everybody can benefit and learn from the conversation.