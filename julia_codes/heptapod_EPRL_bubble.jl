printstyled("\nHeptapod EPRL bubble computation parallelized on $(nworkers()) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 6 && error("please use these 6 arguments: data_sl2cfoam_next_folder    cutoff    shell_min    shell_max     Immirzi    store_folder")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
CUTOFF = parse(Int, ARGS[2])
SHELL_MIN = parse(Int, ARGS[3])
SHELL_MAX = parse(Int, ARGS[4])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[5]))
@eval STORE_FOLDER = $(ARGS[6])

STORE_FOLDER = "$(STORE_FOLDER)/data_folder/EPRL"
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
    conf = SL2Cfoam.Config(VerbosityOff, HighAccuracy, 200, 0)
    SL2Cfoam.cinit(DATA_SL2CFOAM_FOLDER, IMMIRZI, conf)
    # disable C library automatic parallelization
    SL2Cfoam.set_OMP(false)
end
println("done\n")

# logging function (flushing needed)
function log(x...)
    println("[ ", now(), " ] - ", join(x, " ")...)
    flush(stdout)
end

function heptapod_EPRL_bubble(cutoff, shells, store_folder::String)

    # set boundary
    step = onehalf = half(1)

    ampls = Float64[]

    result_return = (ret=true, store=false, store_batches=false)

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a lists with spins to compute
        spins_j12_j15_j25_values = Vector{HalfInt8}[]
        spins_j34_values = Vector{HalfInt8}[]

        for j12::HalfInt = 0:onehalf:pcutoff, j15::HalfInt = 0:onehalf:pcutoff,
            j25::HalfInt = 0:onehalf:pcutoff

            spins_j34 = HalfInt8[]

            for j34::HalfInt = 0:onehalf:pcutoff

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
                push!(spins_j34, j34)

            end

            isempty(spins_j34) && continue

            # must be computed
            push!(spins_j12_j15_j25_values, [j12, j15, j25])
            push!(spins_j34_values, spins_j34)

        end

        if isempty(spins_j12_j15_j25_values)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spins_j12_j15_j25_index in eachindex(spins_j12_j15_j25_values)

            j12 = spins_j12_j15_j25_values[spins_j12_j15_j25_index][1]
            j15 = spins_j12_j15_j25_values[spins_j12_j15_j25_index][2]
            j25 = spins_j12_j15_j25_values[spins_j12_j15_j25_index][3]

            # range of intertwiners
            i1, i1_range = intertwiner_range(j15, onehalf, onehalf, j12)
            i2, i2_range = intertwiner_range(onehalf, onehalf, j25, j15)
            i5, i5_range = intertwiner_range(j12, j25, onehalf, onehalf)

            reduced_range = (i1, i2, (0, 0), (0, 0), i5)

            # compute first EPRL vertex
            v2 = vertex_compute([j15, onehalf, onehalf, j12, onehalf, onehalf, j25, onehalf, onehalf, onehalf], shells, reduced_range; result=result_return)

            df_1 = dim(j12) * dim(j15) * dim(j25)

            amp_1 = 0.0

            for j34 in spins_j34_values[spins_j12_j15_j25_index]

                # compute second EPRL vertex
                v1 = vertex_compute([j12, onehalf, onehalf, j15, onehalf, onehalf, j25, j34, onehalf, onehalf], shells; result=result_return)

                i4, i4_range = intertwiner_range(j34, onehalf, onehalf, onehalf)

                # dim of internal faces
                df_2 = df_1 * sqrt(dim(j34))

                # intertwiner contractions
                amp_2 = 0.0
                @turbo for i1 in 1:i1_range, i2 in 1:i2_range, i4 in 1:i4_range, i5 in 1:i5_range
                    amp_2 += v1.a[i2, i4, i4, i5, i1] * v2.a[i5, 1, 1, i2, i1]
                end

                amp_1 += amp_2 * df_2

            end

            amp_1

        end

        ampl = ampls[end] + tampl

        # comment this line for better performance
        log("Amplitude at partial cutoff = $pcutoff: $(ampl)")

        push!(ampls, ampl)

    end # partial cutoffs loop

    # store partials 
    if (cutoff > 1)
        @save "$(store_folder)/heptapod_ampls_cutoff_$(cutoff)_Immirzi_$(IMMIRZI).jld2" ampls
    end

    ampls

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time heptapod_EPRL_bubble(1, 0, "nothing");
println("done\n")
sleep(1)

ampls_matrix = Array{Float64,2}(undef, 2 * CUTOFF + 1, SHELL_MAX - SHELL_MIN + 1)

printstyled("\nStarting computation with K = $(CUTOFF), Dl_min = $(SHELL_MIN), Dl_max = $(SHELL_MAX), Immirzi = $(IMMIRZI)...\n"; bold=true, color=:cyan)

counter = 0
column_labels = String[]

for Dl = SHELL_MIN:SHELL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:cyan)
    @time ampls = heptapod_EPRL_bubble(CUTOFF, Dl, STORE_FOLDER)
    global counter += 1
    push!(column_labels, "Dl = $(Dl)")
    ampls_matrix[:, counter] = ampls[:]

end

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(ampls_matrix, column_labels)
CSV.write("$(STORE_FOLDER)/heptapod_bubble_cutoff_$(CUTOFF)_Immirzi_$(IMMIRZI)_Dl_min_$(SHELL_MIN)_Dl_max_$(SHELL_MAX).csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
