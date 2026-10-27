# Neurophysical HVAC Control

**Building a controller that can learn the building, not memorize it.**

I’m developing a neural architecture for heating, ventilation and air conditioning in the [Smart Buildings Competition — NeurIPS 2026](https://smart-buildings-2026.github.io/competition/). The question behind this project is simple: **can the same controller make good decisions in a building it has never seen before?**

Buildings have different layouts, equipment and thermal behavior. A useful controller has to work with those differences while keeping people comfortable and accounting for energy cost and carbon emissions.

![Five connected blocks: physical contract, causal state, shared representation, coordinated decisions and constrained actions.](docs/assets/architecture.svg)

## Explore the D2-BASE policy architecture

This repository includes the source of one implemented controller version,
D2-BASE, alongside a small shared-representation module. D2-BASE combines
causal HVAC features, shared neural encoders, zone/AHU/water action heads and a
hard operational projection.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m neurophysical_hvac
python -m unittest discover -s tests -v
```

The included example uses invented feature arrays and untrained random
parameters. The policy source does not include trained weights; it requires a
compatible parameter set to produce actions. The example and tests demonstrate
software structure only, not control performance. This release contains no
quantitative results. [Read the code walkthrough →](docs/code.md)

## Building the architecture, block by block

### 01 · Understand what the building is telling us

Start with the physical meaning of each input: temperatures, comfort bands, equipment signals and the relationships between zones and air-handling units. A temperature setpoint is a request; it is not necessarily the temperature the equipment delivers. Getting that distinction right comes before learning.

### 02 · Give the controller a sense of time

Buildings respond slowly. A single reading cannot describe everything happening inside them. The current controller uses simple causal history, including temperature changes and previous commands. Richer learned memory is a direction I want to test, rather than assume it will help.

### 03 · Learn representations that can be reused

Instead of designing a separate network for each floor plan, share neural encoders across zones and equipment. Combine local information into air-handler and water-system context. This lets the architecture accommodate different numbers of zones; making those representations transfer well is the harder research question.

### 04 · Bring local and shared decisions together

One air handler serves several zones, and their needs can conflict. Cooling, airflow and reheat also affect one another. The architecture connects local zone decisions with shared equipment outputs. I’m investigating which forms of coordination improve control enough to justify their complexity.

### 05 · Turn predictions into usable commands

Translate the network’s outputs into the building’s permitted action interface, apply operational constraints and remember the command actually sent. Valid commands are a starting point; comfort still has to be checked in closed-loop evaluation.

[Explore the architecture →](docs/architecture.md)

## What I’ve built so far

A shared neural baseline, physical feature processing, simple temporal state and constrained action outputs are implemented and have been evaluated locally. These pieces provide a working foundation for the broader five-block design.

Other variants have also been implemented and evaluated locally. This public release does not disclose their outcomes.

The broader architecture is still being developed. Recurrent memory and thermal relationships between neighboring zones remain research ideas. A reinforcement-learning mechanism is at the technical preflight stage; real training rollouts remain future work.

[Progress and lessons →](docs/research-status.md)

## How I decide what belongs in the model

I want each block to earn its place. That means comparing it with a simpler alternative, measuring the controller over complete trajectories and looking at uncomfortable periods as carefully as average performance. Better imitation of another controller does not automatically mean better control.

The policy uses only information available at decision time. Experimental comparisons keep track of provenance and separate development findings from evidence of generalization. [Read the evaluation approach →](docs/evaluation.md)

## Where this is going

The goal is a compact, physically meaningful controller that can adapt to unfamiliar building structures. The next questions are whether better temporal state helps, whether coordination avoids conflicting actions, and whether thermal relationships add useful information beyond equipment connections.

This repository shares the D2-BASE policy source, a representation core and the thinking behind their construction. Trained weights, training implementation, experiment data and quantitative results are not included. The released policy requires externally supplied weights and should not be treated as a useful controller without them. It is an independent project; the simulator and reference interface come from the [official starter kit](https://github.com/smart-buildings-2026/starter-kit). [Sources and attribution →](docs/sources.md)
