### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ 1d804fc2-2d1d-11eb-2637-5b10963d1978
begin
	using PowerSystems;
	using PowerSimulationsDynamics;
	using Sundials;
	using Debugger;
end

# ╔═╡ 6bba4e06-2d11-11eb-034f-f9ca2209f1b4
md"# OMIB Tutorial

Here, model one synchronous machine with no AVR, TG or PSS against an infinite bus"

# ╔═╡ 32f758c8-2d1d-11eb-009a-95738873a0a4
md" ## Data Structure
- `Bus`es
- `Branch`es that connect two buses. Some can be modelled as `DynamicLines`
- `StaticInjection` which are all devices that can inject or withdraw power. Used to solve Power Flow. `DynamicInjection` elements attached to `Static` components to connect PF solution to dynamics of device.
- `PowerLoad` which are all loads connected to buses and also used to solve Power Flow. ConstantPower devices translated to RLC loads for transient simulations.
- `Source` which define source components behind a reactance that inject or withdraw current.
"

# ╔═╡ 3328c06e-2d24-11eb-3fc3-1bdbb8d79c76
md"## One Machine Against Infinite Bus"

# ╔═╡ 8e897a58-2d32-11eb-135b-8f9de3c0b0ac
md"As seem from outline of components below, the OMIB system is composed of:
- System with base power of 100 kW
- Two buses, with Bus 1 at 1.05 p.u. and Bus 2 at 1.05 p.u. with base voltage of 230 V and min/max limits of 0.94 p.u. and 1.06 p.u. for both buses
- Thermal generator producing 50 kW, located at Bus 2 (PV Bus)
- Line connecting Buses 1 and 2 with $X=0.05$ p.u.
"

# ╔═╡ 25301c50-2d24-11eb-2822-5fd8e3cfc3b6
# load data from PTI file
omib_system = System("data/omib_system/OMIB.raw")

# ╔═╡ 3d02aaf0-2d33-11eb-02fc-47a263c214ec
md"### Static System Components

Define buses, lines, generators and sources"

# ╔═╡ 98289760-2d37-11eb-324a-8dfc5c3ef079
collect(get_components(Source, omib_system))

# ╔═╡ 3488d594-2d32-11eb-2104-4db978d2c5c8
collect(get_components(StaticInjection, omib_system))

# ╔═╡ e051a90a-2d32-11eb-022a-6334b30f9dd0
collect(get_components(Bus, omib_system))

# ╔═╡ 1322913c-2d33-11eb-08b7-9305586060cd
collect(get_components(Line, omib_system))

# ╔═╡ d2addc1e-2d33-11eb-21ae-39c1ebde88a6
md"There is no injection device at Bus 1 (slack bus). We will add a `Source` with small impedance and make the bus an *infinite bus*. An infinite bus is one whose voltage and frequency remains constant despite changes in the supply-demand balance. 

An infinite bus can be modelled by many generators operating in parallel"

# ╔═╡ 00141132-2d34-11eb-1e48-4539cca7e33e
begin
	try
		inf_source = Source(name="InfBus", available=true,
							active_power=0.0, reactive_power=0.0,
							bus=collect(get_components(Bus, omib_system))[2],
							R_th=0.0, X_th=5e-6)
		add_component!(omib_system, inf_source)
	catch
		@warn "Infinite bus already added to Bus 1"
	end
end

# ╔═╡ eb692832-2d36-11eb-32a8-59767634083d
md"### Dynamic Components/Injections

These are dynamic components attached to a static component. This includes all components that determine dynamic behaviour. This includes machine, shaft, Automatic Voltage Regulator (AVR), Turbine Governor (TG) and Power System Stabiliser (PSS) Dynamics"

# ╔═╡ 8d8f141a-2d39-11eb-32e9-8f946fff780d
begin
	# Machine
	machine_classic() = BaseMachine(0.0, 0.2995, 0.7087)
	# Shaft
	shaft_damping() = SingleMass(3.148, 2.0)
	# AVR
	avr_none() = AVRFixed(0.0)
	# TG
	tg_none() = TGFixed
	# PSS
	pss_none() = PSSFixed(0.0)
end

# ╔═╡ adad12fa-2d3c-11eb-00b9-034cc9a21190
md"Obtain the Static Injection component from Bus 2"

# ╔═╡ 6d082668-2d3c-11eb-104c-c7c34260726f
static_generator = get_component(Generator, omib_system, "generator-102-1")

# ╔═╡ 665296a0-2d3c-11eb-0078-bdfbe1c4a03b
begin
dyn_gen_classic = DynamicGenerator(static_generator, 1.0, machine_classic(), 									   	   shaft_damping(), avr_none(), tg_none(), 									   		   pss_none())
	try
		add_component!(omib_system, dyn_gen_classic)
	catch
		@warn "Already added dynamic components to the system"
	end
end

# ╔═╡ 40bd1a30-2d3e-11eb-2617-ef59d0fb9914
md"### Data Output"

# ╔═╡ 46d8c2d6-2d3e-11eb-260e-c71bdd9f4f59
begin
	if !isdir("./data/omib_system")
		mkdir("data/omib_system")
	end
	
	to_json(omib_system, "data/omib_system/omib_sys_new.json", force=true)
end

# ╔═╡ 8a12bf18-2d3f-11eb-27cf-115cf5c452fa
md" ### Dynamic Simulation
OMIB has an infinite bus (voltage `Source` behind impedance) and another bus with a classic machine. Simulate a trip of one of two circuits on the line connecting the two, thereby doubling impedance"

# ╔═╡ 219c0052-2d41-11eb-1b48-13d528fc11e2
md"Perturbations can come in two flavours:
- Three Phase Fault
- Change in Reference Parameter

Here, we simulate a three phase fault my modifying the admittance matrix."

# ╔═╡ b09e6526-2d42-11eb-2dcc-09db5562fa32
begin
	fault_branch = deepcopy(get_component(Line, omib_system, "1"))
	# double the impedance of the line
	fault_branch.x = fault_branch.x * 2
	Ybus_fault = Ybus([fault_branch], get_components(Bus, omib_system))[:, :]
	perturb_Ybus = ThreePhaseFault(1.0, Ybus_fault)
end

# ╔═╡ fe2911ce-2dec-11eb-162a-0b3be48548de
begin
	omib_sys_json = System("data/omib_system/omib_sys_new.json")
end

# ╔═╡ facbadec-2d48-11eb-308d-f33570c617c1
begin
	tspan = (0.0, 30.0)
	sim = Simulation("data/omib_system/dynamic", omib_sys_json,
					 tspan, perturb_Ybus)
    print_device_states(sim)
    x0_init = get_initial_conditions(sim)
    execute!(sim, IDA(), dtmax=0.02);
	print(sim.solution)
    # using Plots
    # angle = get_state_series(sim, ("generator-102-1", :ω))
    # plot(angle)
end

# ╔═╡ afcd2796-2d5a-11eb-205a-e39e9408dedf


# ╔═╡ Cell order:
# ╟─6bba4e06-2d11-11eb-034f-f9ca2209f1b4
# ╠═1d804fc2-2d1d-11eb-2637-5b10963d1978
# ╟─32f758c8-2d1d-11eb-009a-95738873a0a4
# ╟─3328c06e-2d24-11eb-3fc3-1bdbb8d79c76
# ╟─8e897a58-2d32-11eb-135b-8f9de3c0b0ac
# ╠═25301c50-2d24-11eb-2822-5fd8e3cfc3b6
# ╟─3d02aaf0-2d33-11eb-02fc-47a263c214ec
# ╠═98289760-2d37-11eb-324a-8dfc5c3ef079
# ╠═3488d594-2d32-11eb-2104-4db978d2c5c8
# ╠═e051a90a-2d32-11eb-022a-6334b30f9dd0
# ╠═1322913c-2d33-11eb-08b7-9305586060cd
# ╟─d2addc1e-2d33-11eb-21ae-39c1ebde88a6
# ╠═00141132-2d34-11eb-1e48-4539cca7e33e
# ╟─eb692832-2d36-11eb-32a8-59767634083d
# ╠═8d8f141a-2d39-11eb-32e9-8f946fff780d
# ╟─adad12fa-2d3c-11eb-00b9-034cc9a21190
# ╠═6d082668-2d3c-11eb-104c-c7c34260726f
# ╠═665296a0-2d3c-11eb-0078-bdfbe1c4a03b
# ╟─40bd1a30-2d3e-11eb-2617-ef59d0fb9914
# ╠═46d8c2d6-2d3e-11eb-260e-c71bdd9f4f59
# ╟─8a12bf18-2d3f-11eb-27cf-115cf5c452fa
# ╟─219c0052-2d41-11eb-1b48-13d528fc11e2
# ╠═b09e6526-2d42-11eb-2dcc-09db5562fa32
# ╠═fe2911ce-2dec-11eb-162a-0b3be48548de
# ╠═facbadec-2d48-11eb-308d-f33570c617c1
# ╠═afcd2796-2d5a-11eb-205a-e39e9408dedf
