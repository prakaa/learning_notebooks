using TimeSeries
using PowerSimulations
using PowerSystems
using DataFrames
using CSV
using Cbc
using Dates
using DataFrames
using Ipopt
using TimeSeries
using PowerSystems
using PowerSimulations

raw_demand = CSV.read("./ReserveSimulations/data/demand.csv", DataFrame)

# create system
sys = System(
    1.0,
)
set_units_base_system!(sys, "NATURAL_UNITS")

# zone
nsw_zone = LoadZone("NSW-LoadZone",
                    maximum(raw_demand[:, :nsw_demand]),
                    0.0
                    )
add_component!(sys, nsw_zone)

# buses
nsw_bus = Bus(1,
              "NSW", 
              "REF",
              nothing,
              nothing,
              nothing,
              nothing,
              nothing,
              nsw_zone,
              )

# generators
bayswater = ThermalStandard(;
    name="Bayswater",
    available=true,
    status=true,
    bus=nsw_bus,
    active_power=2545.0,
    reactive_power=0.0,
    rating=2640.0,
    active_power_limits=(min=1000.0, max=2545.0),
    reactive_power_limits=(min=-1.0, max=1.0),
    ramp_limits=(up=20.67, down=15.33),
    operation_cost=ThreePartCost((0.0, 2.868), 0.0, 200.0, 200.0),
    base_power=1.0,
    time_limits=(up=0.02, down=0.02),
    prime_mover=PrimeMovers.ST,
    fuel=ThermalFuels.COAL,
    )

tallawarra = ThermalStandard(;
    name = "Tallawarra",
    available = true,
    status = true,
    bus = nsw_bus,
    active_power = 395.0,
    reactive_power = 0.0,
    rating = 440.0,
    prime_mover = PrimeMovers.ST,
    fuel = ThermalFuels.NATURAL_GAS,
    active_power_limits = (min = 199.0, max = 395.0),
    reactive_power_limits = (min = -1.0, max = 1.0),
    time_limits = (up = 4.0, down = 4.0),
    ramp_limits = (up = 6.0, down = 6.0),
    operation_cost = ThreePartCost((0.0, 9.338), 0.0, 50.0, 50.0),
    base_power = 100.0,
)

nsw_solar = RenewableFix(;
    name = "NSWSolar",
    available = true,
    bus = nsw_bus,
    active_power = 50.0,
    reactive_power = 0.0,
    rating = 50.0,
    prime_mover = PrimeMovers.PVe,
    power_factor=1.0,
    base_power = 50.0,
)

# loads
nsw_load = PowerLoad(
    "NSWLoad",
    true,
    nsw_bus,
    nothing,
    6900.0,
    0.0,
    1.0,
    maximum(raw_demand[:, :nsw_demand]),
    0.0
)

# reserves
or = StaticReserve{ReserveUp}("OR",true, 300.0, 50.0)

add_components!(sys, [nsw_bus, tallawarra, bayswater, nsw_solar, nsw_load, or])

# add load
date_format = Dates.DateFormat("d/m/y H:M")
di_year = collect(DateTime("01/01/2019 00:05", date_format):
                  Dates.Minute(5):
                  DateTime("01/01/2020 00:00", date_format))

nsw_demand_data = TimeArray(di_year, raw_demand[:, :nsw_demand])
nsw_demand = SingleTimeSeries("max_active_power", nsw_demand_data)
add_time_series!(sys, nsw_load, nsw_demand)

solver = optimizer_with_attributes(Ipopt.Optimizer, "loglevel"=>1, "ratioGap"=>0.5)
ed_problem_template = OperationsProblemTemplate()
set_device_model!(ed_problem_template, ThermalStandard, ThermalDispatch)
set_service_model!(ed_problem_template, StaticReserve{ReserveUp}, RampReserve)
problem = OperationsProblem(ed_problem_template, sys;
                            optimizer=solver, 
                            constraint_duals=[:CopperPlateBalance, 
                                              :requirement__StaticReserve_ReserveUp],
                            horizon=1
                            )
transform_single_time_series!(sys, 1, Minute(5))
sim_problem = SimulationProblems(ED=problem)
sim_sequence = SimulationSequence(
    problems=sim_problem,
    intervals=Dict("ED" => (Minute(5), Consecutive())),
)
sim = Simulation(
    name="economic_dispatch",
    steps=length(di_year),
    problems=sim_problem,
    sequence=sim_sequence,
    simulation_folder="./ReserveSimulations/dispatch_data/"
)

build!(sim)