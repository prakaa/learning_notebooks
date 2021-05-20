using SIIPExamples
pkgpath = dirname(dirname(pathof(SIIPExamples)))
include(joinpath(pkgpath, "test", "3_PowerSimulations_examples", "01_operations_problems.jl"));

using Ipopt
solver = optimizer_with_attributes(Ipopt.Optimizer)

ed_template = template_economic_dispatch(network = ACPPowerModel)

ed_template.devices[:Hydro] = DeviceModel(HydroEnergyReservoir, HydroDispatchRunOfRiver)

problem = OperationsProblem(
    EconomicDispatchProblem,
    ed_template,
    sys,
    horizon = 4,
    optimizer = solver,
    balance_slack_variables = true,
)

solve!(problem)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

