# # Hydropower Simulations with [PowerSimulations.jl](https://github.com/NREL-SIIP/PowerSimulations.jl)

# **Originally Contributed by**: Clayton Barrows and Sourabh Dalvi

# ## Introduction

# PowerSimulations.jl supports simulations that consist of sequential optimization problems
# where results from previous problems inform subsequent problems in a variety of ways.
# This example demonstrates a few of the options for modeling hydropower generation.

# ## Dependencies
using SIIPExamples
pkgpath = dirname(dirname(pathof(SIIPExamples)))

# ### Modeling Packages
using PowerSystems
using PowerSimulations
const PSI = PowerSimulations
using D3TypeTrees

# ### Data management packages
using Dates
using DataFrames

# ### Optimization packages
using Cbc # solver
solver = optimizer_with_attributes(Cbc.Optimizer, "logLevel" => 1, "ratioGap" => 0.05)

# ### Data
# PowerSystems.jl links to some meaningless test data that is suitable for this example.
# The [make_hydro_data.jl](../../script/3_PowerSimulations_examples/make_hydro_data.jl) script
# loads three systems suitable for the examples here.

include(joinpath(pkgpath, "script", "3_PowerSimulations_examples", "make_hydro_data.jl"))

# This line just overloads JuMP printing to remove double underscores added by PowerSimulations.jl
PSI.JuMP._wrap_in_math_mode(str) = "\$\$ $(replace(str, "__"=>"")) \$\$"

# ## Two PowerSimulations features determine hydropower representation.
# There are two principal ways that we can customize hydropower representation in
# PowerSimulations. First, we can play with the formulation applied to hydropower generators
# using the `DeviceModel`. We can also adjust how simulations are configured to represent
# different decision making processes and the information flow between those processes.

# ### Hydropower `DeviceModel`s

# First, the assignment of device formulations to particular device types gives us control
# over the representation of devices. This is accomplished by defining `DeviceModel`
# instances. For hydro power representations, we have two available generator types in
# PowerSystems:

#nb TypeTree(PowerSystems.HydroGen)

# And in PowerSimulations, we have several available formulations that can be applied to
# the hydropower generation devices:

#nb TypeTree(PSI.AbstractHydroFormulation, scopesep = "\n", init_expand = 5)

# Let's see what some of the different combinations create. First, let's apply the
# `HydroDispatchRunOfRiver` formulation to the `HydroEnergyReservoir` generators, and the
# `FixedOutput` formulation to `HydroDispatch` generators.
#  - The `FixedOutput` formulation just acts
# like a load subtractor, forcing the system to accept it's generation.
#  - The `HydroDispatchRunOfRiver` formulation represents the the energy flowing out of
# a reservoir. The model can choose to produce power with that energy or just let it spill by.

devices = Dict{Symbol, DeviceModel}(
    :Hyd1 => DeviceModel(HydroEnergyReservoir, HydroDispatchRunOfRiver),
    :Hyd2 => DeviceModel(HydroDispatch, FixedOutput),
    :Load => DeviceModel(PowerLoad, StaticPowerLoad),
);

template = PSI.OperationsProblemTemplate(CopperPlatePowerModel, devices, Dict(), Dict());

op_problem = PSI.OperationsProblem(GenericOpProblem, template, c_sys5_hy_uc, horizon = 24)

# Now we can see the resulting JuMP model:

op_problem.psi_container.JuMPmodel

# The first two constraints are the power balance constraints that require the generation
# from the controllable `HydroEnergyReservoir` generators to be equal to the load (flat 10.0 for all time periods)
# minus the generation from the `HydroDispatch` generators [1.97, 1.983, ...]. The 3rd and 4th
# constraints limit the output of the `HydroEnergyReservoir` generator to the limit defined by the
# `max_activepwoer` time series. And the last 4 constraints are the lower and upper bounds of
# the `HydroEnergyReservoir` operating range.

#-

# Next, let's apply the `HydroDispatchReservoirBudget` formulation to the `HydroEnergyReservoir` generators.
devices = Dict{Symbol, DeviceModel}(
    :Hyd1 => DeviceModel(HydroEnergyReservoir, HydroDispatchReservoirBudget),
    :Load => DeviceModel(PowerLoad, StaticPowerLoad),
);

template = PSI.OperationsProblemTemplate(CopperPlatePowerModel, devices, Dict(), Dict());

op_problem = PSI.OperationsProblem(GenericOpProblem, template, c_sys5_hy_uc, horizon = 24)

# And, the resulting JuMP model:

op_problem.psi_container.JuMPmodel

# Finally, let's apply the `HydroDispatchReservoirStorage` formulation to the `HydroEnergyReservoir` generators.

devices = Dict{Symbol, DeviceModel}(
    :Hyd1 => DeviceModel(HydroEnergyReservoir, HydroDispatchReservoirStorage),
    :Load => DeviceModel(PowerLoad, StaticPowerLoad),
);

template = PSI.OperationsProblemTemplate(CopperPlatePowerModel, devices, Dict(), Dict());

op_problem = PSI.OperationsProblem(GenericOpProblem, template, c_sys5_hy_uc, horizon = 24)

#-

op_problem.psi_container.JuMPmodel

# ### Multi-Stage `SimulationSequence`
# The purpose of a multi-stage simulation is to represent scheduling decisions consistently
# with the time scales that govern different elements of power systems.

# #### Multi-Day to Daily Simulation:
# In the multi-day model, we'll use a really simple representation of all system devices
# so that we can maintain computational tractability while getting an estimate of system
# requirements/capabilities.

devices = Dict(
    :Generators => DeviceModel(ThermalStandard, ThermalDispatch),
    :Loads => DeviceModel(PowerLoad, StaticPowerLoad),
    :HydroEnergyReservoir =>
        DeviceModel(HydroEnergyReservoir, HydroDispatchReservoirStorage),
)
template_md = OperationsProblemTemplate(CopperPlatePowerModel, devices, Dict(), Dict());

# For the daily model, we can increase the modeling detail since we'll be solving shorter
# problems.

devices = Dict(
    :Generators => DeviceModel(ThermalStandard, ThermalDispatch),
    :Loads => DeviceModel(PowerLoad, StaticPowerLoad),
    :HydroEnergyReservoir =>
        DeviceModel(HydroEnergyReservoir, HydroDispatchReservoirStorage),
)
template_da = OperationsProblemTemplate(CopperPlatePowerModel, devices, Dict(), Dict());

#-

stages_definition = Dict(
    "MD" => Stage(
        GenericOpProblem,
        template_md,
        c_sys5_hy_wk,
        solver,
        system_to_file = false,
    ),
    "DA" => Stage(
        GenericOpProblem,
        template_da,
        c_sys5_hy_uc,
        solver,
        system_to_file = false,
    ),
)

# This builds the sequence and passes the the energy dispatch schedule for the `HydroEnergyReservoir`
# generator from the "MD" stage to the "DA" stage in the form of an energy limit over the
# synchronized periods.

sequence = SimulationSequence(
    step_resolution = Hour(48),
    order = Dict(1 => "MD", 2 => "DA"),
    feedforward_chronologies = Dict(("MD" => "DA") => Synchronize(periods = 2)),
    horizons = Dict("MD" => 2, "DA" => 24),
    intervals = Dict("MD" => (Hour(48), Consecutive()), "DA" => (Hour(24), Consecutive())),
    feedforward = Dict(
        ("DA", :devices, :HydroEnergyReservoir) => IntegralLimitFF(
            variable_source_stage = PSI.ACTIVE_POWER,
            affected_variables = [PSI.ACTIVE_POWER],
        ),
    ),
    cache = Dict(("MD", "DA") => StoredEnergy(HydroEnergyReservoir, PSI.ENERGY)),
    ini_cond_chronology = IntraStageChronology(),
);

#-

file_path = tempdir()

sim = Simulation(
    name = "hydro",
    steps = 1,
    stages = stages_definition,
    stages_sequence = sequence,
    simulation_folder = file_path,
)

build!(sim)

# We can look at the "MD" Model

sim.stages["MD"].internal.psi_container.JuMPmodel

# And we can look at the "DA" model

sim.stages["DA"].internal.psi_container.JuMPmodel

# And we can execute the simulation by running the following command
# ```julia
# sim_results = execute!(sim)
# ```
#-

# #### 3-Stage Simulation:

stages_definition = Dict(
    "MD" => Stage(
        GenericOpProblem,
        template_md,
        c_sys5_hy_wk,
        solver,
        system_to_file = false,
    ),
    "DA" => Stage(
        GenericOpProblem,
        template_da,
        c_sys5_hy_uc,
        solver,
        system_to_file = false,
    ),
    "ED" => Stage(
        GenericOpProblem,
        template_da,
        c_sys5_hy_ed,
        solver,
        system_to_file = false,
    ),
)

sequence = SimulationSequence(
    step_resolution = Hour(48),
    order = Dict(1 => "MD", 2 => "DA", 3 => "ED"),
    feedforward_chronologies = Dict(
        ("MD" => "DA") => Synchronize(periods = 2),
        ("DA" => "ED") => Synchronize(periods = 24),
    ),
    intervals = Dict(
        "MD" => (Hour(48), Consecutive()),
        "DA" => (Hour(24), Consecutive()),
        "ED" => (Hour(1), Consecutive()),
    ),
    horizons = Dict("MD" => 2, "DA" => 24, "ED" => 12),
    feedforward = Dict(
        ("DA", :devices, :HydroEnergyReservoir) => IntegralLimitFF(
            variable_source_stage = PSI.ACTIVE_POWER,
            affected_variables = [PSI.ACTIVE_POWER],
        ),
        ("ED", :devices, :HydroEnergyReservoir) => IntegralLimitFF(
            variable_source_stage = PSI.ACTIVE_POWER,
            affected_variables = [PSI.ACTIVE_POWER],
        ),
    ),
    cache = Dict(("MD", "DA") => StoredEnergy(HydroEnergyReservoir, PSI.ENERGY)),
    ini_cond_chronology = IntraStageChronology(),
);

#-

sim = Simulation(
    name = "hydro",
    steps = 1,
    stages = stages_definition,
    stages_sequence = sequence,
    simulation_folder = file_path,
)

#-

build!(sim)

# We can look at the "MD" Model

sim.stages["MD"].internal.psi_container.JuMPmodel

# And we can look at the "DA" model

sim.stages["DA"].internal.psi_container.JuMPmodel

# And we can look at the "ED" model

sim.stages["ED"].internal.psi_container.JuMPmodel

# And we can execute the simulation by running the following command
# ```julia
# sim_results = execute!(sim)
# ```
#-
