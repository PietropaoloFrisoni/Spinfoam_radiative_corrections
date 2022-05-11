printstyled("\nHeptapod BF bubble computation parallelized on $(nworkers()) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 3 && error("please use this 3 arguments: data_sl2cfoam_next_folder    cutoff    store_folder")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
CUTOFF = parse(Int, ARGS[2])
@eval STORE_FOLDER = $(ARGS[3])

STORE_FOLDER = "$(STORE_FOLDER)/data_folder/BF"
mkpath(STORE_FOLDER)

printstyled("precompiling packages...\n"; bold=true, color=:cyan)
using JLD2
using Distributed
using DataFrames
using Dates
using CSV
@everywhere using SL2Cfoam
@everywhere using HalfIntegers
@everywhere using LoopVectorization
println("done\n")

if (CUTOFF <= 1)
    error("please provide a larger cutoff")
end

isMPI = @ccall SL2Cfoam.clib.sl2cfoam_is_MPI()::Bool
isMPI && error("MPI version not allowed")

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere begin
    Immirzi = 1.200
    conf = SL2Cfoam.Config(VerbosityOff, HighAccuracy, 200, 0)
    SL2Cfoam.cinit(DATA_SL2CFOAM_FOLDER, Immirzi, conf)
    # disable C library automatic parallelization
    SL2Cfoam.set_OMP(false)
end
println("done\n")

# logging function (flushing needed)
function log(x...)
    println("[ ", now(), " ] - ", join(x, " ")...)
    flush(stdout)
end

function heptapod_BF_bubble(cutoff, store_folder::String)

    # set boundary
    step = onehalf = half(1)
    ampls = Float64[]

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a list of all spins to compute
        spins_all = NTuple{4,HalfInt}[]

        for j12::HalfInt = 0:onehalf:pcutoff, j15::HalfInt = 0:onehalf:pcutoff,
            j25::HalfInt = 0:onehalf:pcutoff, j34::HalfInt = 0:onehalf:pcutoff

            # skip if computed in lower partial cutoff
            j12 <= (pcutoff - step) && j15 <= (pcutoff - step) &&
                j25 <= (pcutoff - step) && j34 <= (pcutoff - step) && continue

            # skip if any intertwiner range empty
            r1, _ = intertwiner_range(j12, onehalf, onehalf, j15)
            r2, _ = intertwiner_range(j12, j25, onehalf, onehalf)
            r3, _ = intertwiner_range(onehalf, onehalf, j34, onehalf)
            r4, _ = intertwiner_range(j34, onehalf, onehalf, onehalf)
            r5, _ = intertwiner_range(onehalf, onehalf, j25, j15)

            isempty(r1) && continue
            isempty(r2) && continue
            isempty(r3) && continue
            isempty(r4) && continue
            isempty(r5) && continue

            # must be computed
            push!(spins_all, (j12, j15, j25, j34))

        end

        if isempty(spins_all)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spins in spins_all

            j12, j15, j25, j34 = spins

            # range of intertwiners
            i1, i1_range = intertwiner_range(j12, onehalf, onehalf, j15)
            i2, i2_range = intertwiner_range(j12, j25, onehalf, onehalf)
            i4, i4_range = intertwiner_range(j34, onehalf, onehalf, onehalf)
            i5, i5_range = intertwiner_range(onehalf, onehalf, j25, j15)

            # compute BF vertices
            v1 = vertex_BF_compute([j12, onehalf, onehalf, j15, onehalf, onehalf, j25, j34, onehalf, onehalf])

            # sl2cfoam-next throws an error for K > 30 with reduced range
            # TODO: investigare
            v2 = vertex_BF_compute([j15, onehalf, onehalf, j12, onehalf, onehalf, j25, onehalf, onehalf, onehalf])

            # dim internal faces
            dfj = dim(j12) * dim(j15) * dim(j25) * sqrt(dim(j34))

            # intertwiner contractions
            amp = 0.0
            @turbo for i1 in 1:i1_range, i2 in 1:i2_range, i4 in 1:i4_range, i5 in 1:i5_range
                amp += v1.a[i5, i4, i4, i2, i1] * v2.a[i2, 1, 1, i5, i1]
            end

            amp * dfj

        end

        ampl = ampls[end] + tampl

        log("Amplitude at partial cutoff = $pcutoff: $(ampl)")

        push!(ampls, ampl)

    end # partial cutoffs loop

    # store partials 
    if (cutoff > 1)
        @save "$(store_folder)/heptapod_ampls_cutoff_$(cutoff).jld2" ampls
    end

    ampls

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time ampls = heptapod_BF_bubble(1, "nothing");
println("done\n")
sleep(1)

printstyled("\nStarting computation with K = $(CUTOFF)...\n"; bold=true, color=:cyan)

@time ampls = heptapod_BF_bubble(CUTOFF, STORE_FOLDER);

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)

df = DataFrame(amplitudes=ampls)

CSV.write("$(STORE_FOLDER)/heptapod_bubble_cutoff_$(CUTOFF).csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
