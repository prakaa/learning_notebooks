### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 6bf099a5-48cd-466b-8f83-b4c3adedf7be
begin
	using JuMP
	using DataFrames
	using GLPK
end

# ╔═╡ 658534ae-b901-11eb-3a20-153b9188b496
md"""
# Economic Dispatch (with Reserves)
This is heavily based on the the tutorial available in [JuMP's documentation](https://jump.dev/JuMP.jl/dev/tutorials/Mixed-integer%20linear%20programs/power_systems/).

However, I have tried to adapt the economic dispatch example to demonstrate how reserves (e.g. FCAS) might be co-optimised in economic dispatch and show an example of how co-optimisation incorporates the *opportunity-cost* of reserve provision.
"""

# ╔═╡ 7c118911-43f8-4498-89a7-1ee944b82049
md"""
## Dependencies
"""

# ╔═╡ c5e439bb-434a-4ccb-93c3-45f543dbf86a
md"""
## Hypothetical System

Some major assumptions to simplify this:
- We solve for a single time, so all units of energy expressed in MW
  - No time sequential modelling, which includes ramping constraints
- Assume no transmission losses
- Assume no transmission constraints
- Assume this is a regional, single price market with marginal pricing
- Assume that generator offers reflect short-run marginal costs and that the market accepts only one price-quantity pair (in the NEM, 10 price-quantity pairs are accepted for energy and FCAS)


Furthermore, in terms of generation, we assume:
- Generator 1 is coal and dispatchable
- Generator 2 is CCGT and dispatchable
- There is a third dispatchable generator (OCGT) for the reserve problem
- The wind farms are aggregated into just a single wind unit
"""

# ╔═╡ 72c884ad-18e0-42a6-b615-c5854020ee2f
md"""
![system](https://jump.dev/JuMP.jl/dev/assets/power_systems.png)
"""

# ╔═╡ 6333b70f-6891-45b4-9a2f-7c577ff04cf4
md"""
## Economic Dispatch

Minimise cost of energy supply subject to operational constraints.

``min \sum_{i∈I} c_i^g⋅g_i + c^w⋅w``

subject to:
- ``g_i^{min} ≤ g_i ≤ g_i^{max}`` (min and max power limits)
- ``0 ≤ w ≤ w^f`` (wind power injection ≤ forecast)
- ``∑_{i ∈ I} g_i + w = d^f`` (dispatchable + wind generation = demand forecast)

Where:
- ``g_i`` is the generation output of a dispatchable generator (Generator 1 or 2)

- short-run marginal cost (\$/MW) is given by ``c_i^g``

"""

# ╔═╡ 2a8781db-ec61-4fe7-809f-2d4738fde333
md"""
### JuMP implementation
"""

# ╔═╡ 6e486a98-21b3-4828-a61f-a4ee2e9727a6
function solve_ed(g_max, g_min, c_g, c_w, d, w_f)
	ed = Model(GLPK.Optimizer)
	@variable(ed, g_min[i] <= g[i = 1:2] <= g_max[i]) # dispatchable gen
	@variable(ed, 0 <= w <= w_f) # wind power injection
	@constraint(ed, [i = 1:2], g[i] ≤ g_max[i]) # max power constraint
	@constraint(ed, [i = 1:2], g[i] ≥ g_min[i]) # min power constraint
	@constraint(ed, w <= w_f)
	con = @constraint(ed, sum(g) + w == d) # energy balance
	@objective(ed, Min, c_g' * g + c_w * w)
	optimize!(ed)
	return value.(g), value(w), w_f - value(w), objective_value(ed), ed, con
end

# ╔═╡ 7af097c6-109d-440f-a717-0fae07db5f74
md"""
#### Run ED with input data
"""

# ╔═╡ 85922f92-5d6e-4565-a9c2-2aee77abff9b
begin
	# create input data
	gen_data = DataFrame(g_max = [1000, 1000], g_min = [300, 150],
						 c_g = [50, 100], tech=["coal", "CCGT"])

	gen_data
end

# ╔═╡ f2572466-4ce7-42f8-a84b-44c529b9a72a
begin
  d = 1500
  w_f = 200
  c_w = 20
end;

# ╔═╡ 164852aa-b904-4039-b851-abc40dccd046
md"""

Cost of wind: \$ $(c_w) / MW

Wind forecast: $w_f MW

Demand: $d MW
"""

# ╔═╡ 79b8d28c-3ad7-4726-9d65-955ceaf0f2b7
(g_opt, w_opt, ws_opt, obj, model, con) = solve_ed(gen_data.g_max, gen_data.g_min,
								         	   	   gen_data.c_g, c_w, d, w_f);

# ╔═╡ b026eda0-c5b1-4c63-a755-97fdd06a8203
md"""
Check for feasibility and duality
"""

# ╔═╡ f1b110d5-fed9-4112-8c72-7469f6a80d36
primal_status(model)

# ╔═╡ 31d6d379-497f-4f84-b4bd-46678a0be787
has_duals(model)

# ╔═╡ a22df840-aa3a-4bac-99cc-50bdbacadee8
md"""
Primal feasible and has duals
"""

# ╔═╡ 89f9610b-7495-4790-b0f3-0327a5b6dd65
md"""

#### Results
For a demand of $d MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_opt[1]) MW
  - ``g_2``: $(g_opt[2]) MW
  - Wind: $w_opt MW, spilled = $ws_opt MW

Total cost: \$ $obj

Shadow price of energy balance constraint (supply = demand): $(shadow_price(con)), so marginal cost is \$$(shadow_price(con)*-1)/MW. This is the price of energy.

-----
"""

# ╔═╡ e66ffc80-74e5-43ae-a56d-c94494087410
md"""
#### Reduced demand ED
Now run a model with **demand at 600 MW**.
"""

# ╔═╡ 205fffe4-9cd6-4092-929f-a3ccf0434ff6
md"""
##### A note about modifying JuMP models
If a constraint or the objective function is changed, it is faster to modify specific constraint or objective function to reduce computational burden or rebuilding the model. 

For example, modifying the demand can be done by redefining the constraint and resolving, rather than specifying an entirely new model."""

# ╔═╡ 31843b48-0c7c-487f-ac38-d083e7de7a6a
begin
	d_new = 600
	set_normalized_rhs(con, d_new)
	optimize!(model)
	g_new = value.(model[:g])
	w_new = value(model[:w])
	obj_new = objective_value(model)
end;

# ╔═╡ 3eebdc9b-8132-43b2-96fa-362bee722dee
md"""
#### Results
For a demand of $d_new MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_new[1]) MW
  - ``g_2``: $(g_new[2]) MW
  - Wind: $w_new MW, spilled = $(w_f-w_new) MW

Total cost: \$ $obj_new

Shadow price of energy balance constraint (supply = demand): $(shadow_price(con)), so marginal cost is \$$(shadow_price(con)*-1)/MW. This is the price of energy.

In this case, wind energy is spilled because minimum generation constraints must be met for Generators 1 and 2.

"""

# ╔═╡ 6a158311-f151-40aa-b18c-9254378a6429
md"""
## Economic Dispatch with Reserves

Now we will include reserves into the optimisation problem (co-optimisation). We will assume that only dispatchable generators (Generators 1, 2 and 3) can provide reserves.

Minimise cost of energy supply and reserves subject to operational constraints.

``min \sum_{i∈I} c_i^g⋅g_i + \sum_{i∈I} c_i^r⋅r_i + c^w⋅w``

subject to:
- ``g_i^{min} ≤ g_i`` (reserve and generation must be within capacity limits)
- ``g_i + r_i ≤ g_i^{max}`` (reserve and generation must be within capacity limits)
- ``0 ≤ w ≤ w^f`` (wind power injection ≤ forecast)
- ``∑_{i ∈ I} r_i = R`` (reserve requirement met)
- ``∑_{i ∈ I} g_i + w = d^f`` (dispatchable + wind generation = demand forecast)

Where:
- ``g_i`` is the generation output of a dispatchable generator (Generator 1, 2 or 3)

- short-run marginal cost of energy (\$/MW) is given by ``c_i^g``
- ``r_i`` is the reserve provided by a dispatchable generator (Generator 1, 2 or 3)

- short-run marginal cost of reserve (\$/MW) is given by ``r_i^g``

"""

# ╔═╡ ed843abd-9a1d-442f-913a-473dc6d47bef
md"""
### JuMP Implementation
"""

# ╔═╡ a8a63601-eeef-4c7d-8b43-c651cf4bbbf8
function solve_ed_with_reserves(g_max, g_min, c_g, c_w, c_r, R, d, w_f)
	ed = Model(GLPK.Optimizer)
	@variable(ed, g_min[i] <= g[i = 1:3] <= g_max[i]) # capacity limits
	@variable(ed, g_min[i] <= r[i = 1:3] <= g_max[i]) # capacity limits
	@variable(ed, 0 <= w <= w_f) # wind power injection
	@constraint(ed, [i = 1:3], g[i] + r[i] ≤ g_max[i]) # max power constraint
	@constraint(ed, [i = 1:3], g[i] ≥ g_min[i]) # min power constraint
	@constraint(ed, w <= w_f)
	en_con = @constraint(ed, sum(g) + w == d) # energy balance
	r_con = @constraint(ed, sum(r) == R) # reserve requirement
	@objective(ed, Min, c_g' * g + c_w * w + c_r' * r)
	optimize!(ed)
	return value.(g), value.(r), value(w), objective_value(ed), en_con, r_con, model
end

# ╔═╡ 9a7c3e64-641e-48b8-9c5f-2eb0a1e5727f
begin
	reserve_gen_data = copy(gen_data)
	push!(reserve_gen_data, [500, 0, 300, "OCGT"])
	reserve_gen_data[!, :c_r] = [40, 80, 500]
	reserve_gen_data
end

# ╔═╡ b73b8cc0-bcb6-4a96-ba68-6bd898305760
md"""

Cost of wind: \$ $(c_w) / MW

Wind forecast: $w_f MW

Demand: $d MW

Reserve requirement = 701 MW
- Arbitrary for an interesting example!

"""

# ╔═╡ 882ddb8c-a8eb-42cc-8f53-03723875608a
R = 701

# ╔═╡ be45d9fc-c539-4cde-babd-92c03bc7f044
(g_r, r_r, w_r, obj_r, en_con, r_con, r_mod) =
	solve_ed_with_reserves(reserve_gen_data.g_max,
						   reserve_gen_data.g_min,
						   reserve_gen_data.c_g,
						   c_w, reserve_gen_data.c_r,
						   R, d, w_f);

# ╔═╡ 62d03d73-caa3-430e-8d97-014166a0ac5c
md"""
Check for feasibility and duality
"""

# ╔═╡ a6695f34-24b0-4cf9-915a-1be3ae8ac105
primal_status(r_mod)

# ╔═╡ 3872a97b-5760-4ee3-88b5-adc22a42c34b
has_duals(r_mod)

# ╔═╡ af9e70d2-2fba-436c-8f82-9e8f941074a6
md"""
Primal feasible and has duals
"""

# ╔═╡ 6d70c9f6-d708-4d42-ab23-e449edc6f895
md"""
#### Results
For a demand of $d MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_r[1]) MW, ``r_1``: $(r_r[1]) MW
  - ``g_2``: $(g_r[2]) MW, ``r_2``: $(r_r[2]) MW
  - ``g_3``: $(g_r[3]) MW, ``r_3``: $(r_r[3]) MW
  - Wind: $w_r MW, spilled = $(w_f-w_r) MW

Total cost: \$ $obj_r

Shadow price of energy balance constraint (supply = demand): $(shadow_price(en_con)), so marginal cost is \$$(shadow_price(en_con)*-1)/MW. This is the price of energy.

Shadow price of reserve requirement constraint (∑r = R): $(shadow_price(r_con)), so marginal cost is \$$(shadow_price(r_con)*-1)/MW. This is the price of reserves.

"""

# ╔═╡ 9778a606-ff49-4ce9-927c-c5cd259503e0
md"""
#### Reserve price explanation

Why is the reserve price 280 when reserve offers are 40, 80 or 500?

##### Shadow prices and marginal prices

Marginal prices for energy and reserves are essentially the price to service an infinitesimal increase in demand. For our thought experiment, we will simplify this by thinking of the cost to service the next MW of demand or reserve.

###### Formal and simpler definition
The marginal price for energy is formally the *shadow price* of the supply-demand balance constraint (also known as the value of the Lagrange multiplier, or the value of the dual variable of the constraint at the optimal value of the dual problem). Similarly, the marginal price for reserves is the shadow price of the reserve requirement constraint. In this case, the shadow price represents the additional cost of the objective function if the constraint is relaxed. 

Simply put, a shadow price is effectively the *total additional cost to the system* to supply the next MW of energy or reserve.

##### Marginal price of energy
In the situation above, it is cheapest to service the next MW of energy by turning up Generator 3, so the marginal cost of energy is 300. 

##### Marginal price of reserve
However, for the next MW of reserve, it is actually cheapest to turn Generator 2's energy production down 1 MW and thereby obtain 1 MW of reserve from Generator 2. However, to ensure that energy supply and demand and balanced, Generator 3 must be turned up 1 MW. Obtaining reserve from Generator 2 costs 80 (\$c^r_2\$), turning Generator 2 down 1 MW in energy "costs" -100 (turning it down actually reduces the total cost) and turning Generator 3 up 1 MW in energy costs 300, giving us a total of 280. Hence the *total* cost to the system to increase reserves by 1 MW has been accounted for.

###### Participant's perspective
From the perspective of Generator 2, co-optimisation ensures that opporunity-cost in the energy market is accounted for. Since the price for reserves is set by Generator 2, the marginal price is the sum of its reserve offer cost and its opporunity-cost in the energy market:
1. The reserve offer is 80.
2. By being turned down 1 MW in the energy market, Generator 2 misses out on a profit (opportunity cost). This is:
    - (Price of energy) - (Generator 2's short run marginal cost, or assumed to be anyway), which in this case is (300)-(100) = 200. 
The sum of these is 280, the price of reserves. Simply put, if a unit is backed off energy to provide reserves (e.g. FCAS), the optimisation will ensure that it does not miss out on any profits so long as its bids reflect short run marginal costs.
"""

# ╔═╡ Cell order:
# ╟─658534ae-b901-11eb-3a20-153b9188b496
# ╟─7c118911-43f8-4498-89a7-1ee944b82049
# ╠═6bf099a5-48cd-466b-8f83-b4c3adedf7be
# ╟─c5e439bb-434a-4ccb-93c3-45f543dbf86a
# ╟─72c884ad-18e0-42a6-b615-c5854020ee2f
# ╟─6333b70f-6891-45b4-9a2f-7c577ff04cf4
# ╟─2a8781db-ec61-4fe7-809f-2d4738fde333
# ╠═6e486a98-21b3-4828-a61f-a4ee2e9727a6
# ╟─7af097c6-109d-440f-a717-0fae07db5f74
# ╟─85922f92-5d6e-4565-a9c2-2aee77abff9b
# ╟─f2572466-4ce7-42f8-a84b-44c529b9a72a
# ╟─164852aa-b904-4039-b851-abc40dccd046
# ╠═79b8d28c-3ad7-4726-9d65-955ceaf0f2b7
# ╟─b026eda0-c5b1-4c63-a755-97fdd06a8203
# ╠═f1b110d5-fed9-4112-8c72-7469f6a80d36
# ╠═31d6d379-497f-4f84-b4bd-46678a0be787
# ╟─a22df840-aa3a-4bac-99cc-50bdbacadee8
# ╟─89f9610b-7495-4790-b0f3-0327a5b6dd65
# ╟─e66ffc80-74e5-43ae-a56d-c94494087410
# ╟─205fffe4-9cd6-4092-929f-a3ccf0434ff6
# ╠═31843b48-0c7c-487f-ac38-d083e7de7a6a
# ╟─3eebdc9b-8132-43b2-96fa-362bee722dee
# ╟─6a158311-f151-40aa-b18c-9254378a6429
# ╟─ed843abd-9a1d-442f-913a-473dc6d47bef
# ╠═a8a63601-eeef-4c7d-8b43-c651cf4bbbf8
# ╟─9a7c3e64-641e-48b8-9c5f-2eb0a1e5727f
# ╟─b73b8cc0-bcb6-4a96-ba68-6bd898305760
# ╠═882ddb8c-a8eb-42cc-8f53-03723875608a
# ╠═be45d9fc-c539-4cde-babd-92c03bc7f044
# ╟─62d03d73-caa3-430e-8d97-014166a0ac5c
# ╠═a6695f34-24b0-4cf9-915a-1be3ae8ac105
# ╠═3872a97b-5760-4ee3-88b5-adc22a42c34b
# ╟─af9e70d2-2fba-436c-8f82-9e8f941074a6
# ╟─6d70c9f6-d708-4d42-ab23-e449edc6f895
# ╟─9778a606-ff49-4ce9-927c-c5cd259503e0
