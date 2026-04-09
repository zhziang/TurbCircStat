using CUDA, ProgressMeter, HDF5
include("utils.jl")

#Parse argument
aps = ArgParseSettings()
@add_arg_table! aps begin
	"--npoints", "-n"
	help = "Number of point vortices (even number, half positive halfnegative)."
	arg_type = Int
	default = 10
	"--nsamples", "-s"
	help = "Number of samples."
	arg_type = Int
	default = 100
	"--path", "-p"
	help = "The path of the output file."
	arg_type = String
	default = @__DIR__
end
args = parse_args(aps)
npoints = args["npoints"]
nsamples = args["nsamples"]

Γ = 2 .* [CUDA.ones(npoints÷2); -CUDA.ones(npoints÷2)] ./ npoints

bin = range(-1/npoints, 1/npoints, 101)
hist = zeros(length(bin) - 1)

@showprogress for _ in 1:nsamples
	idx = floor(Int, (hamiltonian(CUDA.rand(npoints), CUDA.rand(npoints), Γ) - first(bin))/step(bin)) + 1
	if 1 ≤ idx ≤ length(bin)-1
		hist[idx] += 1
	end
end

h5open((@__DIR__) * "/entropy_N$(npoints).h5","w") do fid
    fid["state_density"] = Array(hist)
    fid["hamiltonian_level"] = Array((bin[1:end-1] + bin[2:end]) / 2)
end
