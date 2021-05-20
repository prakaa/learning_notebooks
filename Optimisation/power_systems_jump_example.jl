### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 6bf099a5-48cd-466b-8f83-b4c3adedf7be
begin
	using JuMP
	using DataFrames
	using GLPK
	using PlutoUI
end

# ╔═╡ 658534ae-b901-11eb-3a20-153b9188b496
md"""
# JuMP Power Systems Tutorial
Based on the tutorial available [here](https://jump.dev/JuMP.jl/dev/tutorials/Mixed-integer%20linear%20programs/power_systems/) and based on a Ross Baldick paper. The example mirrors ERCOT market - West vs East.
"""

# ╔═╡ 7c118911-43f8-4498-89a7-1ee944b82049
md"""
## Dependencies
"""

# ╔═╡ c5e439bb-434a-4ccb-93c3-45f543dbf86a
md"""
## Texas System
"""

# ╔═╡ 72c884ad-18e0-42a6-b615-c5854020ee2f
md"""
![system](https://jump.dev/JuMP.jl/dev/assets/power_systems.png)
"""

# ╔═╡ 6333b70f-6891-45b4-9a2f-7c577ff04cf4
md"""
## Economic Dispatch

Minimise cost of energy supply subject to operational constraints.

min ``\sum_{i∈I} c_i^g⋅g_i + c^w⋅w``

subject to:
- ``g_i^{min} ≤ g_i ≤ g_i^{max}`` (min and max power limits)
- ``0 ≤ w ≤ w^f`` (wind power injection ≤ forecast)
- ``∑_{i ∈ I} g_i + w = d^f`` (dispatchable + wind generation = demand forecast)

Where:
- ``g_i`` is the generation output of a dispatchable generator (Generator 1 or 2)

- short-run marginal cost ($/MWH) is given by ``c_i^g``

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
						 c_g = [50, 100], c_fixed = [1000, 0])

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

Cost of wind: \$ $(c_w) / MWh

Wind forecast: $w_f MW

Demand: $d MW
"""

# ╔═╡ 79b8d28c-3ad7-4726-9d65-955ceaf0f2b7
(g_opt, w_opt, ws_opt, obj, model, con) = solve_ed(gen_data.g_max, gen_data.g_min,
								         	   	   gen_data.c_g, c_w, d, w_f);

# ╔═╡ 89f9610b-7495-4790-b0f3-0327a5b6dd65
md"""

For a demand of $d MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_opt[1]) MW
  - ``g_2``: $(g_opt[2]) MW
  - Wind: $w_opt MW, spilled = $ws_opt MW

Total cost: \$ $obj

Shadow price of energy balance constraint: $(shadow_price(con)), so marginal cost is \$$(shadow_price(con)*-1)/MWh

-----
"""

# ╔═╡ e66ffc80-74e5-43ae-a56d-c94494087410
md"""

Now run a model with demand at 1400 MW.
"""

# ╔═╡ 205fffe4-9cd6-4092-929f-a3ccf0434ff6
md"""
If a constraint or the objective function is changed, it is faster to modify specific constraint or objective function to reduce computational burden or rebuilding the model. 

For example, modifying the demand can be done by redfining the constraint and resolving, rather than specifying an entirely new model."""

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

For a demand of $d_new MW, economic dispatch results in a dispatch of:
  - ``g_1``: $(g_new[1]) MW
  - ``g_2``: $(g_new[2]) MW
  - Wind: $w_new MW, spilled = $(w_f-w_new) MW

Total cost: \$ $obj_new

Shadow price of energy balance constraint: $(shadow_price(con)), so marginal cost is \$$(shadow_price(con)*-1)/MWh

"""

# ╔═╡ 19dead1b-858b-424a-90fb-860a06dce79e
md"""
### Limitations of current implementation
#### Wind spillage
- Does not perform unit commitment
- As demand is reduced, min. generation levels for unit 1 and 2 will be maintained despite wind being cheaper
#### Transmission constraints
  - Market based and does not consider limitations of transmissions
  - In NEM:
    - Interconnector constraints implemented
    - Optimal power flow used to determine constraints for ED
    - Regional rather than nodal
"""

# ╔═╡ 337c5d57-768a-4cba-8f79-87fd36e9e32a
md"""
## Unit Commitment

We introduce binary variables (1 for synchronised and dispatchable, 0 for not synchronised).

In the model, we can achieve this by modifying the constraint for unit energy generation:
  - ``g_i^{min} ⋅ u_{t, i} ≤ g_i ≤ g_i^{max} ⋅ u_{ti, i}`` where ``u_i ∈ {0,1}``

``t`` here is important as ``u`` could be 0 at time ``t`` and 1 at time ``t+1``. This would require start-up (and in the converse situation,  shut-down) costs to be account for.
"""

# ╔═╡ 6e6bbea5-8500-49b3-aad1-fd4eb6c8706c
md"""
### JuMP implementation
"""

# ╔═╡ 384e5354-95fb-47b6-8173-155d177f8009
function solve_uc(g_max, g_min, c_g, c_w, d, w_f)
	uc = Model(GLPK.Optimizer)
	@variable(uc, 0 <= g[i=1:2] <= g_max[i]) # gen between 0 and max
	@variable(uc, u[i = 1:2], Bin) # binary variables
	@variable(uc, 0 <= w <= w_f) # wind power injection
	@constraint(uc, [i = 1:2], g[i] <= g_max[i] * u[i]) # max constraint with integer
	@constraint(uc, [i = 1:2], g[i] >= g_min[i] * u[i]) # min constraint with integer
	@constraint(uc, w <= w_f) # wind forecast constraint
	@constraint(uc, sum(g) + w == d) # energy balance
	@objective(uc, Min, c_g'* g + c_w * w)
	optimize!(uc)
	status = termination_status(uc)
	if status != MOI.OPTIMAL
		return status, zeros(length(g)), 0.0, 0.0, zeros(length(u)), Inf
    end
    return status, value.(g), value(w), w_f - value(w), value.(u), 		
		   objective_value(uc), uc
end

# ╔═╡ 148b413c-2f76-4d97-a1c9-644484d825c1
md"""
### Run Unit Commitment

Run with a low demand of 600 MW (as per second ED model).
"""

# ╔═╡ 9c3ca595-9e11-43ea-a599-95905c5812fc
(status, g_uc, w_uc, ws_uc, bin, obj_uc, uc) = solve_uc(gen_data.g_max, 
														gen_data.g_min,
								            	   		gen_data.c_g, c_w, d_new, 
														w_f);

# ╔═╡ 26207350-9622-47a8-9e5c-32649ba85dbc
status

# ╔═╡ 4f8c9187-d8e9-48a3-a3d6-f86876288ddb
primal_status(uc)

# ╔═╡ 427dd841-d66a-4fe9-b86d-bad5a6030ff6
dual_status(uc)

# ╔═╡ 56d11661-6bad-4e73-afff-7d31dd638501
md"""
No dual problem, hence cannot calculate shadow price
"""

# ╔═╡ c138be43-ad56-4212-bcee-f1b311bd3e98
md"""

For a demand of $d_new MW, unit commitment results in commitment:
  - ``g_1``: $(bin[1]) - $(g_uc[1]) MW
  - ``g_2``: $(bin[2]) - $(g_uc[2]) MW
  - Wind: $w_uc MW, spilled = $(ws_uc) MW

Total cost: \$ $obj_uc

Saving compared to economic dispatch: \$$(obj_new-obj_uc)
"""

# ╔═╡ cbe76c5a-79fe-423d-b6da-6ef985d0f4c4
md"""
We have not run time sequential unit commitment or economic dispatch to account for ramping constraints and start-up and shut-down costs
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
# ╠═f2572466-4ce7-42f8-a84b-44c529b9a72a
# ╟─164852aa-b904-4039-b851-abc40dccd046
# ╠═79b8d28c-3ad7-4726-9d65-955ceaf0f2b7
# ╟─89f9610b-7495-4790-b0f3-0327a5b6dd65
# ╟─e66ffc80-74e5-43ae-a56d-c94494087410
# ╟─205fffe4-9cd6-4092-929f-a3ccf0434ff6
# ╠═31843b48-0c7c-487f-ac38-d083e7de7a6a
# ╟─3eebdc9b-8132-43b2-96fa-362bee722dee
# ╟─19dead1b-858b-424a-90fb-860a06dce79e
# ╟─337c5d57-768a-4cba-8f79-87fd36e9e32a
# ╟─6e6bbea5-8500-49b3-aad1-fd4eb6c8706c
# ╠═384e5354-95fb-47b6-8173-155d177f8009
# ╟─148b413c-2f76-4d97-a1c9-644484d825c1
# ╠═9c3ca595-9e11-43ea-a599-95905c5812fc
# ╟─26207350-9622-47a8-9e5c-32649ba85dbc
# ╠═4f8c9187-d8e9-48a3-a3d6-f86876288ddb
# ╠═427dd841-d66a-4fe9-b86d-bad5a6030ff6
# ╟─56d11661-6bad-4e73-afff-7d31dd638501
# ╟─c138be43-ad56-4212-bcee-f1b311bd3e98
# ╟─cbe76c5a-79fe-423d-b6da-6ef985d0f4c4
