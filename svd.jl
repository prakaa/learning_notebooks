### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 4c53eb50-2585-11eb-081b-6343823a23c2
begin
	using Images, TestImages
	pic = Gray.(load("./images/mandrill.jpg"))
end

# ╔═╡ 47ca9f9a-2588-11eb-3fe1-6f88a7d5b1d3
begin
	using LinearAlgebra
	function svd_truncate(img, r::Int)
		U, Σ, V = svd(img)
		# return first r cols, rxr sub-block and first r cols transposed
		M = U[:, 1:r] * Diagonal(Σ[1:r]) * V[:, 1:r]'
		return M
	end
end

# ╔═╡ 2ff0c884-258a-11eb-2161-4574cbc5d1fe
begin
	using Plots
	_, S, _ = svd(pic)
	plot(S; label="Singular values")
end

# ╔═╡ 7b908984-26ee-11eb-03c1-c164e61adc0d
begin
	using CSV, DataFrames
	wines = CSV.read("./data/winequality-red.csv", DataFrame)
end

# ╔═╡ f915c12c-257d-11eb-3b14-8b8e1b2d88c3
md"# Singular Value Decompostion"

# ╔═╡ 29d43546-257e-11eb-1798-61d73166af62
md"$X=UΣV^*$
where U and V are unitary, which means $UU^*=U^*U=I$, and have orthonormal columns
and where $Σ$ has real, non-negative entries on the diagonal. This can also be represented using the economy SVD:

$X=\hat{U}\hat{Σ}V^*$
where $U = [\hat{U} \hat{U}^⟂]$ and $\hat{U}^⟂$ spans a space complementary and orthogonal to $\hat{U}$, $\hat{Σ}$ contains the non-zero entries of $Σ$. This form is more efficient for non-square matrices.

Remember, $V^*$ means complex conjugate transpose, or where real, just the transpose"

# ╔═╡ 26cba85c-257e-11eb-3dc0-551d61207a7d
md"## Truncation

$X \approx \tilde{U}\tilde{Σ}\tilde{V^*}$ 
where for $\tilde{X}$, $\tilde{U}$ and $\tilde{V^*}$, the first r leading columns of U and V are taken and for $\tilde{Σ}$ is the leading $r×r$ sub-block. This can also be written as a dyadic summation, since it's just diagonals multiplied by the left and right singular vectors:

$\tilde{X}=\Sigma_{k=1}^{r}\sigma_ku_kv^*_k+...+\sigma_ru_rv^*_r$

For a given rank $r$, this is the best approximation for $X$"

# ╔═╡ 35886928-2585-11eb-3d3a-c5ea32f5f1dc
md"### Image Compression"

# ╔═╡ 93d6a062-2594-11eb-0abf-5da6e912a558
plot(cumsum(S)./sum(S); label="Cumulative energy for r modes")

# ╔═╡ b8cc2bd0-2594-11eb-0f1c-11b1b01406a1
begin
	truncated_pics = [Gray.(svd_truncate(pic, r)) for r in range(10, 100, step=10)]
	mosaicview(truncated_pics, nrow=2, ncol=5)
end

# ╔═╡ 73228b5a-2635-11eb-29c8-c7e7200b98fd
md"## Multiple Linear Regression"

# ╔═╡ ef3b62d0-26f2-11eb-2067-9393a36af2a6
md"### Moore-Penrose Pseudoinverse:
$A = \tilde{U}\tilde{Σ}\tilde{V^*}$
Moore Penrose PSI:

$A^+ = \tilde{V}\tilde{Σ}^{-1}\tilde{U^*}$

In an overdetermined system, where A∈C^{(n,m)} where n<<m, least-squares soltuion determined by:

$x = A^+Ax = A^+b = \tilde{V}\tilde{Σ}^{-1}\tilde{U^*}b$

We can just use `pinv`
"

# ╔═╡ b46074f0-26ef-11eb-07f7-3349bbba759b
begin
	A = wines[:, 1:11]
	b = wines[:, 12]
	# pad A matrix with ones for constant term
	rows = size(wines, 1)
	A = [A ones(rows)]
	# compute left Moore Penrose PSI
	N = pinv(convert(Matrix, A))
	xtilde = N * convert(Vector, b)
end

# ╔═╡ 756f6d2a-26f2-11eb-3edc-e35dc3a0d008
begin
	plot(wines[:quality], label="actual")
	plot!(convert(Matrix, A) * xtilde, label="regression")
end

# ╔═╡ d608b018-26f4-11eb-0d46-a983f066603c
begin
	plot(sort(wines[:quality]), label="actual - sorted")
	plot!(sort(convert(Matrix, A) * xtilde), label="regression - sorted")
end

# ╔═╡ Cell order:
# ╟─f915c12c-257d-11eb-3b14-8b8e1b2d88c3
# ╟─29d43546-257e-11eb-1798-61d73166af62
# ╟─26cba85c-257e-11eb-3dc0-551d61207a7d
# ╟─35886928-2585-11eb-3d3a-c5ea32f5f1dc
# ╠═4c53eb50-2585-11eb-081b-6343823a23c2
# ╠═47ca9f9a-2588-11eb-3fe1-6f88a7d5b1d3
# ╟─2ff0c884-258a-11eb-2161-4574cbc5d1fe
# ╟─93d6a062-2594-11eb-0abf-5da6e912a558
# ╠═b8cc2bd0-2594-11eb-0f1c-11b1b01406a1
# ╟─73228b5a-2635-11eb-29c8-c7e7200b98fd
# ╟─ef3b62d0-26f2-11eb-2067-9393a36af2a6
# ╠═7b908984-26ee-11eb-03c1-c164e61adc0d
# ╠═b46074f0-26ef-11eb-07f7-3349bbba759b
# ╠═756f6d2a-26f2-11eb-3edc-e35dc3a0d008
# ╠═d608b018-26f4-11eb-0d46-a983f066603c
