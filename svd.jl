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
	function svd_truncate(img::Array, r::Int)
		"""
		Returns an approximation of img truncated to rank-r using
		singular value decomposition
		"""
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
	red_wines = CSV.read("./data/winequality-red.csv", DataFrame)
end

# ╔═╡ 39643d90-27da-11eb-3d9b-77f408dacc4c
# load white wines data, add type column and remove quality columns
begin
	using StatsBase
	white_wines = CSV.read("./data/winequality-white.csv", DataFrame)
	# add type columns
	insertcols!(white_wines, "type" => "white")
	insertcols!(red_wines, "type"=>"red")
	# remove quality columns
	white_wines_attr = white_wines[:, Not(:quality)]
	red_wines_attr = red_wines[:, Not(:quality)]
	wine_attrs = [white_wines_attr[1:size(red_wines_attr,1),:]; red_wines_attr]
end

# ╔═╡ 33f62696-27dc-11eb-3f6c-1748defef375
begin
	using Statistics
	X = convert(Matrix, wine_attrs[:, Not(:type)])
	# row-wise mean	
	X_til = ones(1, size(X, 1))' * mean(X, dims=1)
	# compute mean subtracted data
	B = X - X_til
	# compute svd
	U, Σ, Vt = svd(B)
	size(Vt)
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
# plot relative cumulative energy
plot(cumsum(S)./sum(S); label="Cumulative energy for r modes")

# ╔═╡ b8cc2bd0-2594-11eb-0f1c-11b1b01406a1
begin
	truncated_pics = [Gray.(svd_truncate(pic, r)) for r in range(10, 100, step=10)]
	mosaicview(truncated_pics, nrow=2, ncol=5, rowmajor=true)
end

# ╔═╡ 73228b5a-2635-11eb-29c8-c7e7200b98fd
md"## Multiple Linear Regression"

# ╔═╡ ef3b62d0-26f2-11eb-2067-9393a36af2a6
md"### Moore-Penrose Pseudoinverse:
$A = \tilde{U}\tilde{Σ}\tilde{V^*}$
Moore Penrose PSI:

$A^+ = \tilde{V}\tilde{Σ}^{-1}\tilde{U^*}$

In an overdetermined system, where A∈$C^{(n,m)}$ where n<<m, least-squares solution determined by: 

$x = A^+Ax = A^+b = \tilde{V}\tilde{Σ}^{-1}\tilde{U^*}b$

We can also just use `pinv` to calculate the PSI.
"

# ╔═╡ 6312960e-27d6-11eb-395e-a7acc8011c97
md"#### Example - Qualty of Red Wine

Can various features be used to determine what influences red wine quality?
"

# ╔═╡ b46074f0-26ef-11eb-07f7-3349bbba759b
begin
	A = red_wines[:, 1:11]
	b = red_wines[:, 12]
	# pad A matrix with ones for constant term
	rows = size(red_wines, 1)
	A = [A ones(rows)]
	# compute left Moore Penrose PSI
	N = pinv(convert(Matrix, A))
	xtilde = N * convert(Vector, b)
end

# ╔═╡ 756f6d2a-26f2-11eb-3edc-e35dc3a0d008
begin
	plot(red_wines[:, :quality], label="actual")
	plot!(convert(Matrix, A) * xtilde, label="regression")
end

# ╔═╡ d608b018-26f4-11eb-0d46-a983f066603c
begin
	plot(sort(red_wines[:, :quality]), label="actual - sorted")
	plot!(sort(convert(Matrix, A) * xtilde), label="regression - sorted")
end

# ╔═╡ ac155e7c-27d6-11eb-0619-e517d0ff58a9
md"## Principal Component Analysis

Given a matrix $X$ with measurements across variables along a row and measurements over time down a column (i.e. row-based entries):

1. Compute row-wise mean (mean of all rows) $\bar{x}_j=\frac{1}{n}Σ_{i=1}^{n}x_{i,j}$
2. Subtract mean from data. $B=X-\bar{X}$ where $\bar{X}=[1\cdots]^T\bar{x}$ (i.e. project mean of column across all rows in that column)
3. Find the underlying patterns that maximise covariance matrix $C=\frac{1}{n-1}B^*B$. This is done by finding the eigenvector with the largest eigenvalue. Based on the proof around dominant correlations, $\tilde{V}$ contains eigenvectors of $B^*B$, with eigenvalues given by $\tilde{Σ}$, assuming economy SVD. If ordered by magnitude of singular values, columns of U represent decreasing proportions of variances as you go left to right.
4. Score matrix, or coordinates of rows of B mapped to principal component directions $T=BV$.
5. Cumulative energy $g_j=\Sigma_{k=1}^{j} D_{kk}$

Where data is entered with columns as measurements and measurements over space or time as rows:
1. Compute column-wise mean
2. Score matrix $T=\tilde{U}^*x$
3. Principal components contained within $\tilde{U}$

Data can be approximated: $\tilde{x}_{test}=\tilde{U}\tilde{U}^*x_{test}$
"

# ╔═╡ b8e5cbf2-27d9-11eb-24ac-0b4ece146b32
md" #### Example - Wine Clusters

Can we differentiate red and white wine based on some principal components?"

# ╔═╡ 9fee3c02-27da-11eb-3b7e-b122473c1684
md"We want to find PCs, so need to find eigenvectors of correlation matrix. Each row is a data point, so:
1. We compute the row-wise mean.
2. Compute the mean-subtracted data $B$.
3. Run SVD, then extract eigenvalues ($σ^2$) and eigenvectors from $\tilde{V}$.
4. Project rows across PCs."


# ╔═╡ 2503f7f4-27e0-11eb-322e-7775422d4435
plot(cumsum(Σ)/sum(Σ), label="cumulative energy")

# ╔═╡ e3f0ad3c-27e1-11eb-00ce-656462b95762
md"Looks like >95% of variance is captured within first 3 principal components. We can project the data on these first three principal components"

# ╔═╡ a04bd5ba-27e2-11eb-0154-0319b6850b54
begin
	function calculate_xyz(B, Vt, wine_type)
		"""
		Calculates x, y and z values for each data row in the 
		first three principal component directions if the wine type
		is the same as that specified in the arguments
		"""
		xs, ys, zs = Array{Float64}(undef, 1), 
					 Array{Float64}(undef, 1),
					 Array{Float64}(undef, 1)
		# go through each row and transform to first three PCs
		# add color to color series
		for i in range(1, size(B, 1), step=1)
			data_row = B[i, :]
			x = dot(data_row, Vt'[:, 1])
			y = dot(data_row, Vt'[:, 2])
			z = dot(data_row, Vt'[:, 3])
			data_wine_type = wine_attrs[i, :type]
			if data_wine_type == wine_type
				push!(xs, x)
				push!(ys, y)
				push!(zs, z)
			end
		end
		return xs, ys, zs
	end
end

# ╔═╡ 4172f568-27e3-11eb-3f7e-532ed1353b9d
begin
	x_red, y_red, z_red = calculate_xyz(B, Vt, "red")
	x_white, y_white, z_white = calculate_xyz(B, Vt, "white")
	p1 = scatter(x_red, y_red, color=:red, 
			  label="Red Wine", alpha=0.5, markersize=2)
	p1 = scatter!(x_white, y_white, color=:yellow,
			 label="White Wine", alpha=0.5, markersize=2)
	p2 = scatter(x_white, y_white, z_white, color=:yellow,
				 label="White Wine", alpha=0.5, markersize=2)
	p2 = scatter!(x_red, y_red, z_red, color=:red,
				 label="Red Wine", alpha=0.5, markersize=2)
	plot(p1, p2, layout=2)
	xlabel!("PC 1")
	ylabel!("PC 2")
end

# ╔═╡ Cell order:
# ╟─f915c12c-257d-11eb-3b14-8b8e1b2d88c3
# ╟─29d43546-257e-11eb-1798-61d73166af62
# ╟─26cba85c-257e-11eb-3dc0-551d61207a7d
# ╟─35886928-2585-11eb-3d3a-c5ea32f5f1dc
# ╠═4c53eb50-2585-11eb-081b-6343823a23c2
# ╠═47ca9f9a-2588-11eb-3fe1-6f88a7d5b1d3
# ╠═2ff0c884-258a-11eb-2161-4574cbc5d1fe
# ╠═93d6a062-2594-11eb-0abf-5da6e912a558
# ╠═b8cc2bd0-2594-11eb-0f1c-11b1b01406a1
# ╟─73228b5a-2635-11eb-29c8-c7e7200b98fd
# ╟─ef3b62d0-26f2-11eb-2067-9393a36af2a6
# ╟─6312960e-27d6-11eb-395e-a7acc8011c97
# ╠═7b908984-26ee-11eb-03c1-c164e61adc0d
# ╠═b46074f0-26ef-11eb-07f7-3349bbba759b
# ╠═756f6d2a-26f2-11eb-3edc-e35dc3a0d008
# ╠═d608b018-26f4-11eb-0d46-a983f066603c
# ╟─ac155e7c-27d6-11eb-0619-e517d0ff58a9
# ╟─b8e5cbf2-27d9-11eb-24ac-0b4ece146b32
# ╠═39643d90-27da-11eb-3d9b-77f408dacc4c
# ╟─9fee3c02-27da-11eb-3b7e-b122473c1684
# ╠═33f62696-27dc-11eb-3f6c-1748defef375
# ╠═2503f7f4-27e0-11eb-322e-7775422d4435
# ╟─e3f0ad3c-27e1-11eb-00ce-656462b95762
# ╠═a04bd5ba-27e2-11eb-0154-0319b6850b54
# ╠═4172f568-27e3-11eb-3f7e-532ed1353b9d
