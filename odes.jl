### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ c9e964e4-2861-11eb-26b2-39cd0e6cccad
using DifferentialEquations, Plots

# ╔═╡ 73d302ac-2935-11eb-392e-6158ef4b4326
begin
	using StaticArrays
	using BenchmarkTools
end

# ╔═╡ 3a7a04bc-285c-11eb-331f-c13925eb279f
md"# Ordinary Differential Equations and DifferentialEquations.jl"

# ╔═╡ 5656b978-285c-11eb-2132-f32f184e7995
md"## ODE Classifications

Explicit ODE:
$F(x,y,y'⋯,y^{(n-1)})=y^{(n)}$

Implicit ODE:
$F(x,y,y'⋯,y^{(n-1)})=0$

Linear ODE:
$y^{(n)}=Σ_{i=0}^{n-1}a_i(x)y^{(i)} + r(x)$
* where $r(x)=0$, ODE is homogenous as trivial solution y=0
* where $r(x)\neq 0$
"

# ╔═╡ 1a5f36f6-285d-11eb-28bd-4d992ae6ba77
md"## First Order ODE

Defined as $x'=f(x,p,t)$ including IVP $y(t_0)=u_0$
"

# ╔═╡ 0b5933ee-285d-11eb-01e0-293ef69d0927
md"### Example: Exponential Growth"

# ╔═╡ 50e80946-2862-11eb-1912-2598567483e8
md"$\frac{dx}{dt}=0.98x$

Analytical solution is:
$x(t) = e^{0.98t}e^C = a_0e^{0.98t} = 1.0e^{0.98t}$
"

# ╔═╡ 710f8546-285d-11eb-23bf-238c4072b2e3
# exponential increase - 98% increase per year
f(x, p, t) = 0.98x

# ╔═╡ 8bf07c62-285d-11eb-0fe6-d1a092bb7ba7
# look at time t=0 to t=10 and solve ODE
begin
	t = (0.0, 10.0)
	x0 = 1.0
	prob = ODEProblem(f, x0, t)
end

# ╔═╡ b0ebe52a-2861-11eb-2ddf-6396dbeca146
begin
	sol = solve(prob, abstol=1e-8, reltol=1e-8, saveat=0.1)
	# plot true solution u(t)=u0exp(at)
	plot(sol.t, t->1.0*exp(0.98.*t), linewidth=2,
		  xaxis="Time", yaxis="x(t)", 
		  label="Analytical Solution", linestyle=:dash)
	#plot ODE solution
	plot!(sol.t, sol.u, linewidth=2, xaxis="Time",
		 yaxis="x(t)", label="Solution")
	xlims!(0, 5)
	ylims!(1,20)
end

# ╔═╡ 51ed7a0c-288e-11eb-2c7e-0fca95eb6c32
md"### Continuous output
`sol` is a continuous function, so can call as a function of time e.g. `sol(0.45)`


### Key params for `solve`
1. `saveat`: solver save at specific times
2. `dense`: boolean, turns off continuous output
3. `reltol` and `abstol`
4. `alg_hints`: list of Symbols, e.g. [:stiff], gives hints about which alg to use
"

# ╔═╡ b0ac57e2-2925-11eb-28ef-9ff9f4d0c733
md" ## System of ODES - Lorenz"

# ╔═╡ ba4597b4-2925-11eb-2e0d-81e318822d5d
md" 
$\frac{dx}{dt} = σ(y-x)$

$\frac{dy}{dt} = x(ρ-z)-y$

$\frac{dz}{dt} = xy-βz$
"

# ╔═╡ 295c6e6a-292e-11eb-2d1e-43bdb3e151ce
md" We can write this in the in-place format, where $d\textbf{u}$, which contains all derivatives, is an input to the function and where its elements are defined within the function. 

If defined out of place, i.e. `[dx,dy,dz]` as output, array made each time the function is called.
"

# ╔═╡ 69b04830-288f-11eb-32e2-5b8fc717e37f
#define lorenz function as a function of du, u, parameters p and timesteps t
function lorenz!(du,u,p,t)
    σ,ρ,β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

# ╔═╡ 442caa26-2890-11eb-259f-83b3cf9f095d
begin
	# initial conditions
	u0 = [1.0, 0.0, 0.0]
	p = [10, 28, 8/3]
	tspan = (0.0, 100.0)
	prob_lorenz = ODEProblem(lorenz!, u0, tspan, p)
end

# ╔═╡ d18ea752-2926-11eb-187b-d7e73da37ac1
begin
	# i x j matrix, where i is the time i and j is the value of variable j
	sol_lorenz = solve(prob_lorenz)
	varplot = plot(sol_lorenz)
	threevarplot = plot(sol_lorenz, vars=(1,2,3))
	threevarnointer = plot(sol_lorenz, vars=(1,2,3), denseplot=false)
	plot(varplot, threevarplot, threevarnointer, layout=(3,1), size=(700,900))
end

	
	

# ╔═╡ 547bc5b6-2928-11eb-1544-7d0bcb0e9b41
md"## Types
- Is you use `Float64` for IC, then `solve` will respect that type
- Best to change abstol and reltol, and to have time as same type
"

# ╔═╡ 2e1b302e-2929-11eb-2479-b171d1ff2853
md"## Stiffness
* Stiffness is when Jacobian has large eigenvalues
* Or big numbers in $f$

This will result in large changes in the original function or vertical shifts.

Standard algorithms will be slow to solve, e.g. Tsit5() which is Runge Kutta pairs of order 5. 

### Choosing methods
- Higher order, lower tolerance. Lower order, higher tolerance
	- `Tsit5()` is non-stiff Runge Kutta order 5
	- `Vern7/9()` for low tolerances
	- `BS3` for hihg tolerances
- Runge-Kutta are best for non-stiff, Rosenbruck best for small stiff, BDF best for large stiff
	- `Rosenbrock23()`, `Rodas5()`, or `CVODE_BDF()` for stiff
- Can time using BenchmarkTools

"

# ╔═╡ c29faae0-292e-11eb-04c2-9db2216ade96
md"## Optimisation
### Small systems
- Define in-place, i.e. dx or du as input to function, with elements being the system of equations
- [Heap](http://net-informations.com/faq/net/stack-heap.htm) vs stack allocations - more efficient to use stack allocated for small systems ($\lt$ 100 variables). `StaticArray` from StaticArrays.jl is useful. When doing this, use out-of-place allocating form
### Large systems
- **Temporary allocations** are an issue for larger systems, i.e. A+B+C does A+B then (A+B)+C. Broadcast fusion enables addition at the same time. Can use the following: `map((a,b,c)->a+b+c,A,B,C)` or `A.+B.+C` or `@. A+B+C`.
- Allocation of output is also an issue.
- Above two items solved by mutation, where an original variable is chaged. `test!(D,A,B,C) = @. D = A + B + C` allocates output to one array `D`, and this mutation enables values inside D to be changed. Any function with `!` has a mutating form
"

# ╔═╡ 5157e2ca-292f-11eb-02e9-f54603094363
begin
	function lorenz_static(u,p,t)
		σ, ρ, β = p
		dx = σ*(u[2]-u[1])
		dy = u[1]*(ρ-u[3])-u[2]
		dz = u[1]*u[2]-β*u[3]
		#out-of-place allocation
		@SVector [dx, dy, dz]
	end
	# supply initial conditions as SVector
	u0_stat = @SVector [1.0, 0.0, 0.0]
	tspan_stat = (0.0, 100.0)
	p_stat = [10.0, 28.0, 8/3]
	lor_stat_prob = ODEProblem(lorenz_static, u0_stat, tspan_stat, p_stat)
end

# ╔═╡ d1f6140a-2935-11eb-2cda-0d3dc9735a0b
@benchmark solve(prob_lorenz, Tsit5())

# ╔═╡ 50edff70-2936-11eb-3afb-4baac8546f3e
@benchmark solve(lor_stat_prob, Tsit5())

# ╔═╡ 0ac9c24a-293b-11eb-18e9-2f0410ff3b40
md"## Events and Callbacks
- Event triggered when system reaches a state, callbacks handle events
- Callbacks can also preserve conservation laws

### Continuous
- Condition will trigger when = 0 according to interpolation
### Discrete
- Will check after each step if condition true
- e.g. Kid kicking ball at `t=2`

### Sets of Callbacks
Callbacks can be merged with `CallbackSet`
- Only ContinuousCallback that triggers at earliest time is used if multiple callbacks trigger
- Then DiscreteCallbacks used in order

### Callback library
- `ManifoldProjection`: manifold such that $g(sol)=0$, e.g. conservation of energy
- `SavingCallback`: Save values based on function taking `u`,`t` and `integrator` as inputs
"

# ╔═╡ 1ed61612-2945-11eb-1e1d-fdce473cc01f
md"### Continuous Example"

# ╔═╡ 1afcd9b6-293d-11eb-2c0d-37cea484ed36
begin
	# looks at first element of u, when y=0
	function condition(u,t,integrator)
		u[1]
	end
	# make ball bouce, with small friction constant damping velocity
	function affect!(integrator)
		integrator.u[2] = -integrator.p[2] * integrator.u[2]
	end
	bounce = ContinuousCallback(condition, affect!)
end

# ╔═╡ 27554ada-2945-11eb-1f0d-41145fd43263
md"### Discrete Example"

# ╔═╡ 2efdbeaa-2945-11eb-21ff-53ecd4bd9712
begin
	# condition when t=2
	function condition_kick(u, t, integrator)
		t == 2
	end
	# affect, kid kicking ball
	function affect_kick!(integrator)
		integrator.u[2] += 50
	end
end

# ╔═╡ 55b2d55c-2946-11eb-18ab-4f7596850a76
md"### Integration Termination and Directions
- Can terminate integration if a condition is met"

# ╔═╡ 76d7e574-2946-11eb-28f4-919b7d79acf0
harmonic! = @ode_def HarmonicOscillatorODE begin
	dv_hm = -x
	dx = v_hm
end

# ╔═╡ 957cce46-2947-11eb-2539-6fdf418036e2
md"Note that the integrator only terminates at the upcrossing"

# ╔═╡ 5c0c59e0-2947-11eb-1fdc-87f26cc97c8b
begin
	function terminate_affect!(integrator)
		terminate!(integrator)
	end
	
	function terminate_condition(u,t,integrator)
		u[2]
	end
	
	# terminate callbck only trigger on upcrossings
	terminate_cb = ContinuousCallback(terminate_condition, terminate_affect!, nothing)
	
	tspan_hm = (0.0, 10.0)
	u0_hm = [1.0, 0.0]
	prob_hm = ODEProblem(harmonic!,u0_hm,tspan_hm)
	sol_hm = solve(prob_hm, callback=terminate_cb)
	plot(sol_hm)
end

# ╔═╡ 6ed042ca-2948-11eb-201b-476dc68c6198
md"### Manifold Projection

Integrator is drifting over a long period, energy is accumulating

Manifold Callback means energy set back to zero if constraints ever violated"

# ╔═╡ 7c638028-2948-11eb-37f1-5d473bb262ef
begin
	prob_hm_drift = ODEProblem(harmonic!,u0_hm,(0.0,10000.0))
	sol_hm_drift = solve(prob_hm_drift)
	v_plot = plot(sol_hm_drift, vars=(0,1), label="Velocity")
	energy_plot = plot(sol_hm_drift.t, [u[2]^2 + u[1]^2 for u in sol_hm_drift.u],
					   label="Energy")
	plot(v_plot, energy_plot)
end

# ╔═╡ 2a67951a-2949-11eb-3868-c3293e10c5b9
begin
	# define function g to use in manifold
	# number of conditions must be same as size of system
	function g(resid, u, p, t)
		resid[1] = u[2]^2 + u[1]^2 - 1
		resid[2] = 0
	end
	
	man_cb = ManifoldProjection(g)
	sol_hm_con = solve(prob_hm_drift, callback=man_cb)
	v_plot_con = plot(sol_hm_con, vars=(0,1), label="Velocity")
	energy_plot_con = plot(sol_hm_con.t, [u[2]^2 + u[1]^2 for u in sol_hm_con.u],
						   label="Energy Conserved")
	plot(v_plot_con, energy_plot_con)
end

# ╔═╡ b4ebc894-293c-11eb-18be-079a0f44be0c
ball! = @ode_def BallBounceODE begin
  dy =  v
  dv = -g
end g

# ╔═╡ 16c68fe8-2944-11eb-13b8-5baa19e5a563
begin
	u0_ball = [50.0, 0.0]
	tspan_ball = (0.0, 5.0)
	p_ball = (9.8, 0.9)
	prob_ball = ODEProblem(ball!,u0_ball,tspan_ball,p_ball,callback=bounce)
	sol_ball = solve(prob_ball, Tsit5())
	plot(sol_ball)
end

# ╔═╡ 75829aee-2945-11eb-0b74-137b903875d8
begin
	kick_cb = DiscreteCallback(condition_kick, affect_kick!)
	prob_ball_kick = ODEProblem(ball!,u0_ball,tspan_ball,p_ball,callback=kick_cb)
	# solver needs to stop or step exactly at t=2 to effect callback
	sol_ball_kick = solve(prob_ball_kick, Tsit5(), tstops=[2.0])
	plot(sol_ball_kick)
end

# ╔═╡ ae7ea1f8-294a-11eb-32f6-97190d444dd9
md"## Plotting
- Defined with the `@ode_def` macro, can plot using `vars` using Symbol of variable name
- Otherwise can use indices, where 0 is t and 1,2,3... are other vars
-`denseplot=true` interpolates, and `plotdensity` controls number of points
"

# ╔═╡ 2abe7b34-2951-11eb-29c4-039644945b57
md"## Other Cool Features in Notebooks
- in SciMLTutorials.jl: Bayesian inference for params, Monte Carlo based on IC, uncertainty propagation
- Unitful.jl for units
"

# ╔═╡ Cell order:
# ╟─3a7a04bc-285c-11eb-331f-c13925eb279f
# ╠═c9e964e4-2861-11eb-26b2-39cd0e6cccad
# ╟─5656b978-285c-11eb-2132-f32f184e7995
# ╟─1a5f36f6-285d-11eb-28bd-4d992ae6ba77
# ╟─0b5933ee-285d-11eb-01e0-293ef69d0927
# ╟─50e80946-2862-11eb-1912-2598567483e8
# ╠═710f8546-285d-11eb-23bf-238c4072b2e3
# ╠═8bf07c62-285d-11eb-0fe6-d1a092bb7ba7
# ╠═b0ebe52a-2861-11eb-2ddf-6396dbeca146
# ╟─51ed7a0c-288e-11eb-2c7e-0fca95eb6c32
# ╟─b0ac57e2-2925-11eb-28ef-9ff9f4d0c733
# ╟─ba4597b4-2925-11eb-2e0d-81e318822d5d
# ╟─295c6e6a-292e-11eb-2d1e-43bdb3e151ce
# ╠═69b04830-288f-11eb-32e2-5b8fc717e37f
# ╠═442caa26-2890-11eb-259f-83b3cf9f095d
# ╠═d18ea752-2926-11eb-187b-d7e73da37ac1
# ╟─547bc5b6-2928-11eb-1544-7d0bcb0e9b41
# ╟─2e1b302e-2929-11eb-2479-b171d1ff2853
# ╟─c29faae0-292e-11eb-04c2-9db2216ade96
# ╠═73d302ac-2935-11eb-392e-6158ef4b4326
# ╠═5157e2ca-292f-11eb-02e9-f54603094363
# ╠═d1f6140a-2935-11eb-2cda-0d3dc9735a0b
# ╠═50edff70-2936-11eb-3afb-4baac8546f3e
# ╟─0ac9c24a-293b-11eb-18e9-2f0410ff3b40
# ╟─1ed61612-2945-11eb-1e1d-fdce473cc01f
# ╠═b4ebc894-293c-11eb-18be-079a0f44be0c
# ╠═1afcd9b6-293d-11eb-2c0d-37cea484ed36
# ╠═16c68fe8-2944-11eb-13b8-5baa19e5a563
# ╟─27554ada-2945-11eb-1f0d-41145fd43263
# ╠═2efdbeaa-2945-11eb-21ff-53ecd4bd9712
# ╠═75829aee-2945-11eb-0b74-137b903875d8
# ╟─55b2d55c-2946-11eb-18ab-4f7596850a76
# ╠═76d7e574-2946-11eb-28f4-919b7d79acf0
# ╟─957cce46-2947-11eb-2539-6fdf418036e2
# ╠═5c0c59e0-2947-11eb-1fdc-87f26cc97c8b
# ╟─6ed042ca-2948-11eb-201b-476dc68c6198
# ╠═7c638028-2948-11eb-37f1-5d473bb262ef
# ╠═2a67951a-2949-11eb-3868-c3293e10c5b9
# ╟─ae7ea1f8-294a-11eb-32f6-97190d444dd9
# ╠═2abe7b34-2951-11eb-29c4-039644945b57
