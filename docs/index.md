# wildfire-sim

A cellular-automaton wildfire spread model with physically scaled ignition
probabilities, modelled suppression resources, and black-box optimization of
reactive suppression policies over Monte Carlo ensembles. APC 524 final project.

- [Installation](installation.md)
- [Usage](usage.md)
- [Project plan](plan.md) - model design, data sources, milestones and work packages
- [Lattice study](lattice_study.md) - what each neighbour stencil can and cannot represent
- [Performance](performance.md) - profile, engine timings and ensemble costs
- [API Reference](api.md)

## Documented Functions

Fully documented functions in `wildfire_sim.spread.rothermel` (parameters, return values, exceptions raised and usage examples):

- [`load_fuel_models`](api.md#wildfire_sim.spread.rothermel.load_fuel_models): read and validate the fuel model table
- [`fuel_bed`](api.md#wildfire_sim.spread.rothermel.fuel_bed): no-wind, no-slope Rothermel fuel bed values
- [`surface_fire`](api.md#wildfire_sim.spread.rothermel.surface_fire): head-fire spread rate, intensity, and flame length on a grid
