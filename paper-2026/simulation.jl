using GeophysicalFlows, Random, CUDA, HDF5, ArgParse, ProgressMeter

# Parse argument
aps = ArgParseSettings()
@add_arg_table! aps begin
	"--re"
	help = "The Reynolds number of the simulation."
	arg_type = Float64
	default = 4.0
	"--ngrid", "-n"
	help = "Simulation resolution."
	arg_type = Int
	default = 128
	"--ndata"
	help = "Number of frames to be stored."
	arg_type = Int
	default = 400
	"--nstep"
	help = "Number of steps between frames."
	arg_type = Int
	default = 10
	"--cfl"
	help = "CFL number defined as,(ε/k_f)^(1/3)(Δt/Δx)."
	arg_type = Float64
	default = 0.1
	"--gpu"
	help = "Using GPU acceleration."
	action = :store_true
	"--path", "-p"
	help = "The path of the output file."
	arg_type = String
	default = @__DIR__
end
args = parse_args(aps)
Re = args["re"]
ngrid = args["ngrid"]
ndata = args["ndata"]
nstep = args["nstep"]
cfl = args["cfl"]
dev = args["gpu"] ? GPU() : CPU()

# Other constants
ε = 1.0        # Energy injection rate
L = 2π         # domain size
kf = ngrid / (3 * sqrt(Re))       # Forcing wavenumber 
ν = ε^(1 / 3) * kf^(-4 / 3) / Re
dt = (L / ngrid) * (ε / kf)^(-1 / 3) * cfl


# Get the forcing spectrum
forcing_spectrum = begin
	forcing_wavenumber = kf * 2π / L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
	forcing_bandwidth = 1.5 * 2π / L  # the width of the forcing spectrum, `δ_f`

	grid = TwoDGrid(CPU(); nx = ngrid, Lx = L)

	K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber

	forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))

	forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

	ε0 = FourierFlows.parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
	forcing_spectrum .*= ε / ε0        # normalize forcing to inject energy at rate ε	
	device_array(dev)(forcing_spectrum)
end

# Functions describing the random forcing
function calcF!(Fh, sol, t, clock, vars, params, grid)
	randn!(Fh)
	@. Fh *= sqrt(forcing_spectrum) / sqrt(clock.dt)
	return nothing
end

# Instantiate the TwoDNavierStokes problem
prob = TwoDNavierStokes.Problem(dev;
	nx = ngrid,
	Lx = L,
	ν,
	dt,
	stepper = "ETDRK4",
	calcF = calcF!,
	stochastic = true,
)

# Initialize out put file & get the initial value.
output_path = args["path"] * "/.output/Re$(Re)_N$(ngrid).h5"
diag_path = args["path"] * "/output/Re$(Re)_N$(ngrid)_diag.h5"
isdir(args["path"] * "/.output/") || mkdir(args["path"] * "/.output/")
isdir(args["path"] * "/output/") || mkdir(args["path"] * "/output/")
if isfile(output_path)
	sol = h5open(output_path, "r") do f
		frames = [parse(Int, k) for k in keys(f)]
		read(f, "$(maximum(frames))")
	end
	ζ₀ = prob.grid.rfftplan \ device_array(dev)(sol)
	set_ζ!(prob, ζ₀)
else
	set_ζ!(prob, device_array(dev)(zeros(ngrid, ngrid)))
end

# Main loop
fid = h5open(output_path, "w")
fid_diag = h5open(diag_path, "w")
diag = create_dataset(fid_diag, "energy", Float64, (ndata))
cfln = create_dataset(fid_diag, "cfl number", Float64, (ndata))

@showprogress for nframe in 1:ndata
	stepforward!(prob, nstep)
	if any(isnan.(prob.sol))
		error("prob.sol = NaN. Terminated.")
	end
	write_dataset(fid, "$(nframe)", Array(prob.sol))
	E = FourierFlows.parsevalsum(abs2.(prob.sol) .* prob.grid.invKrsq, prob.grid)
    diag[nframe] = E
	cfln[nframe] = dt * sqrt(E) * ngrid / L
end
close(fid_diag)
close(fid)
