using Distributed

number_of_workers = nworkers()
number_of_processes = nprocs()
number_of_threads = Threads.nthreads()
available_cpus = length(Sys.cpu_info())

printstyled("\nGoat EPRL divergence parallelized on $(number_of_workers) worker(s) and $(number_of_threads) thread(s)\n\n"; bold=true, color=:blue)

if (number_of_workers * number_of_threads > available_cpus)
    printstyled("WARNING: you are using more resources than available cores on this system. Performances will be affected\n\n"; bold=true, color=:red)
end

length(ARGS) < 6 && error("please use these 6 arguments: data_sl2cfoam_next_folder    cutoff    shell_min    shell_max     Immirzi    store_folder")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
SHELL_MIN = parse(Int, ARGS[3])
SHELL_MAX = parse(Int, ARGS[4])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[5]))
@eval STORE_FOLDER = $(ARGS[6])

printstyled("precompiling packages...\n"; bold=true, color=:cyan)
@everywhere begin
    include("pkgs.jl")
    include("init.jl")
end
println("done\n")

CUTOFF_FLOAT = parse(Float64, ARGS[2])
CUTOFF = HalfInt(CUTOFF_FLOAT)

if (CUTOFF <= 1)
    error("please provide a larger cutoff")
end

STORE_FOLDER = "$(STORE_FOLDER)/data/EPRL/immirzi_$(IMMIRZI)/divergence/cutoff_$(CUTOFF_FLOAT)"
mkpath(STORE_FOLDER)

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)
println("done\n")

# not (yet) optimized
function goat_EPRL(cutoff, shells)

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]

    result_return = (ret=true, store=false, store_batches=false)

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a list of bulk spins to compute
        bulk_spins_pcutoff = NTuple{4,HalfInt}[]

        for j45::HalfInt = 0:onehalf:pcutoff, j34::HalfInt = 0:onehalf:pcutoff,
            j23::HalfInt = 0:onehalf:pcutoff, j25::HalfInt = 0:onehalf:pcutoff

            # skip if computed in lower partial cutoff
            j25 <= (pcutoff - step) && j34 <= (pcutoff - step) &&
                j23 <= (pcutoff - step) && j45 <= (pcutoff - step) && continue

            # skip if any intertwiner range empty
            i1, _ = intertwiner_range(jb, j25, j25, jb)
            i2, _ = intertwiner_range(j34, jb, j25, jb)
            i3, _ = intertwiner_range(jb, j23, j25, jb)
            i4, _ = intertwiner_range(jb, j45, j25, jb)

            isempty(i1) && continue
            isempty(i2) && continue
            isempty(i3) && continue
            isempty(i4) && continue

            # must be computed
            push!(bulk_spins_pcutoff, (j25, j34, j23, j45))

        end

        if isempty(bulk_spins_pcutoff)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for bulk_spins in bulk_spins_pcutoff

            j25, j34, j23, j45 = bulk_spins

            # range of intertwiners
            i1, i1_range = intertwiner_range(jb, j25, j25, jb)
            i2, i2_range = intertwiner_range(j34, jb, j25, jb)
            i3, i3_range = intertwiner_range(jb, j23, j25, jb)
            i4, i4_range = intertwiner_range(jb, j45, j25, jb)
            reduced_range = (i1, (0, 0), i2, i2, (0, 0))

            # compute EPRL vertices
            v1 = vertex_compute([jb, j25, j25, jb, jb, jb, jb, j34, jb, jb], shells, reduced_range; result=result_return)
            v2 = vertex_compute([jb, j25, j25, jb, jb, j23, j25, jb, j45, jb], shells; result=result_return)

            # dim internal faces
            dfj = dim(j25) * dim(j34) * dim(j45) * dim(j23)

            # intertwiner contractions
            amp = 0.0

            for i1 in 1:i1_range, i3 in 1:i3_range, i2 in 1:i2_range, i4 in 1:i4_range
                amp += v1.a[1, i2, i2, 1, i1] * v2.a[i4, i3, i4, i3, i1]
            end

            amp * dfj * (-1)^(2*j23)

        end

        # if-else for integer spin case
        if isempty(ampls)
            ampl = tampl
            log("Amplitude at partial cutoff = $pcutoff: $(ampl)")
            push!(ampls, ampl)
        else
            ampl = ampls[end] + tampl
            log("Amplitude at partial cutoff = $pcutoff: $(ampl)")
            push!(ampls, ampl)
        end

    end # partial cutoffs loop

    ampls

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time goat_EPRL(1, 0);
println("done\n")
sleep(1)


ampls_matrix = Array{Float64,2}(undef, convert(Int, 2 * CUTOFF + 1), SHELL_MAX - SHELL_MIN + 1)

printstyled("\nStarting computation with K = $(CUTOFF), Dl_min = $(SHELL_MIN), Dl_max = $(SHELL_MAX), Immirzi = $(IMMIRZI)...\n"; bold=true, color=:cyan)

column_labels = String[]

for Dl = SHELL_MIN:SHELL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)
    @time ampls = goat_EPRL(CUTOFF, Dl)
    push!(column_labels, "Dl = $(Dl)")
    ampls_matrix[:, Dl-SHELL_MIN+1] = ampls[:]

end

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(ampls_matrix, column_labels)
CSV.write("$(STORE_FOLDER)/goat_Dl_min_$(SHELL_MIN)_Dl_max_$(SHELL_MAX).csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
