using CUDA, ProgressMeter, HDF5, ArgParse
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

circs = [CUDA.ones(npoints÷2); -CUDA.ones(npoints÷2)] ./ npoints

function hamiltonian(u)
    x, y = mod.(u[:, 1], 1), mod.(u[:, 2], 1)
    xdiff, ydiff = x .- transpose(x), y .- transpose(y)
    H = circs .* green.(xdiff, ydiff) .* transpose(circs)

    H[1:npoints.==(1:npoints)'] .= 0.0
    return sum(H)
end

bin = range(-1/2npoints, 1/2npoints, 101)
hist = zeros(length(bin) - 1)

@showprogress for _ in 1:nsamples
	idx = floor(Int, (hamiltonian(CUDA.rand(npoints,2)) - first(bin))/step(bin)) + 1
	if 1 ≤ idx ≤ length(bin)-1
		hist[idx] += 1
	end
end

h5open((@__DIR__) * "/entropy_N$(npoints).h5","w") do fid
    fid["state_density"] = Array(hist)
    fid["hamiltonian_level"] = Array((bin[1:end-1] + bin[2:end]) / 2)
end
