### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ c9e964e4-2861-11eb-26b2-39cd0e6cccad
using DifferentialEquations, Plots

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
# plot true solution u(t)=u0exp(at)
	plot!(sol.t, t->1.0*exp(0.98.*t), linewidth=2,
		  xaxis="Time", yaxis="x(t)", 
		  label="Analytical Solution")

# ╔═╡ 69b04830-288f-11eb-32e2-5b8fc717e37f
function lorenz!(du,u,p,t)
    σ,ρ,β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

# ╔═╡ 442caa26-2890-11eb-259f-83b3cf9f095d


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
# ╠═51ed7a0c-288e-11eb-2c7e-0fca95eb6c32
# ╠═69b04830-288f-11eb-32e2-5b8fc717e37f
# ╠═442caa26-2890-11eb-259f-83b3cf9f095d
