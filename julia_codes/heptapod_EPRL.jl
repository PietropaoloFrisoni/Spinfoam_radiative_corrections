using Distributed

number_of_workers = nworkers()
number_of_processes = nprocs()
number_of_threads = Threads.nthreads()
available_cpus = Sys.cpu_info()

printstyled("\nHeptapod EPRL divergence parallelized on $(number_of_workers) worker(s) and $(number_of_threads) thread(s)\n\n"; bold=true, color=:blue)

if (number_of_workers*number_of_threads > length(available_cpus))
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

function heptapod_EPRL(cutoff, shells)

    number_of_threads = Threads.nthreads()

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]

    result_return = (ret=true, store=false, store_batches=false)

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a lists with spins to compute
        spin_j45_j34_j35_pcutoff = Vector{HalfInt8}[]
        spin_j78_pcutoff = Vector{HalfInt8}[]

        for j45::HalfInt = 0:onehalf:pcutoff, j34::HalfInt = 0:onehalf:pcutoff,
            j35::HalfInt = 0:onehalf:pcutoff

            spin_j78 = HalfInt8[]

            for j78::HalfInt = 0:onehalf:pcutoff

                # skip if computed in lower partial cutoff
                j45 <= (pcutoff - step) && j34 <= (pcutoff - step) &&
                j35 <= (pcutoff - step) && j78 <= (pcutoff - step) && continue

                # skip if any intertwiner range empty
                i7, _ = intertwiner_range(jb, jb, jb, j78)
                i5, _ = intertwiner_range(j45, j35, jb, jb)
                i4, _ = intertwiner_range(j34, jb, jb, j45)
                i3, _ = intertwiner_range(jb, jb, j35, j34)
                i8, _ = intertwiner_range(j78, jb, jb, jb)

                isempty(i7) && continue
                isempty(i5) && continue
                isempty(i4) && continue
                isempty(i3) && continue
                isempty(i8) && continue

                # must be computed
                push!(spin_j78, j78)

            end

            isempty(spin_j78) && continue

            # must be computed
            push!(spin_j45_j34_j35_pcutoff, [j45, j34, j35])
            push!(spin_j78_pcutoff, spin_j78)

        end

        if isempty(spin_j45_j34_j35_pcutoff)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spin_index in eachindex(spin_j45_j34_j35_pcutoff)

            j45 = spin_j45_j34_j35_pcutoff[spin_index][1]
            j34 = spin_j45_j34_j35_pcutoff[spin_index][2]
            j35 = spin_j45_j34_j35_pcutoff[spin_index][3]

            # range of intertwiners
            i5, i5_range = intertwiner_range(j45, j35, jb, jb)
            i4, i4_range = intertwiner_range(j34, jb, jb, j45)
            i3, i3_range = intertwiner_range(jb, jb, j35, j34)
            reduced_range = (i4, i5, (0, 0), (0, 0), i3)

            # compute first EPRL vertex
            v1 = vertex_compute([j45, jb, jb, j34, jb, jb, j35, jb, jb, jb], shells, reduced_range; result=result_return)

            # dim internal faces
            dfj = dim(j45) * dim(j34) * dim(j35)

            ampt = zeros(number_of_threads)

            Threads.@threads for j78 in spin_j78_pcutoff[spin_index]

                # compute second EPRL vertex
                v2 = vertex_compute([j34, jb, jb, j45, jb, jb, j35, j78, jb, jb], shells; result=result_return)

                i7, i7_range = intertwiner_range(jb, jb, jb, j78)

                amp_2 = 0.0

                @inbounds for i7 in 1:i7_range, i3 in 1:i3_range, i4 in 1:i4_range, i5 in 1:i5_range
                    amp_2 += v1.a[i3, 1, 1, i5, i4] * v2.a[i5, i7, i7, i3, i4]
                end

                ampt[Threads.threadid()] += amp_2 * dim(j78)

            end

            amp = sum(ampt) 

            amp * dfj

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
@time heptapod_EPRL(1, 0);
println("done\n")
sleep(1)

ampls_matrix = Array{Float64,2}(undef, convert(Int,2 * CUTOFF + 1), SHELL_MAX - SHELL_MIN + 1)

printstyled("\nStarting computation with K = $(CUTOFF), Dl_min = $(SHELL_MIN), Dl_max = $(SHELL_MAX), Immirzi = $(IMMIRZI)...\n"; bold=true, color=:cyan)

column_labels = String[]

for Dl = SHELL_MIN:SHELL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)
    @time ampls = heptapod_EPRL(CUTOFF, Dl)
    push!(column_labels, "Dl = $(Dl)")
    ampls_matrix[:, Dl-SHELL_MIN+1] = ampls[:]

end

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(ampls_matrix, column_labels)
CSV.write("$(STORE_FOLDER)/heptapod_Dl_min_$(SHELL_MIN)_Dl_max_$(SHELL_MAX).csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
