using Distributed

number_of_workers = nworkers()
number_of_processes = nprocs()
number_of_threads = Threads.nthreads()
available_cpus = length(Sys.cpu_info())

printstyled("\nFrogface EPRL divergence parallelized on $(number_of_workers) worker(s) and $(number_of_threads) thread(s)\n\n"; bold=true, color=:blue)

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

function frogface_EPRL(cutoff, shells)

    number_of_threads = Threads.nthreads()

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]

    result_return = (ret=true, store=false, store_batches=false)

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate lists with spins to compute
        spin_j25_j34_pcutoff = Vector{HalfInt8}[]
        spin_j23_pcutoff = Vector{HalfInt8}[]
        spin_j45_pcutoff = Vector{Vector{HalfInt8}}[]

        for j25::HalfInt = 0:onehalf:pcutoff, j34::HalfInt = 0:onehalf:pcutoff

            spin_j23 = HalfInt8[]
            spin_j45_ext = Vector{HalfInt8}[]

            for j23::HalfInt = 0:onehalf:pcutoff

                spin_j45 = HalfInt8[]

                for j45::HalfInt = 0:onehalf:pcutoff

                    # skip if computed in lower partial cutoff
                    j25 <= (pcutoff - step) && j34 <= (pcutoff - step) &&
                        j23 <= (pcutoff - step) && j45 <= (pcutoff - step) && continue

                    # skip if any intertwiner range empty
                    i1, _ = intertwiner_range(jb, jb, j45, j34)
                    i2, _ = intertwiner_range(j34, jb, j25, j34)
                    i4, _ = intertwiner_range(j34, j23, jb, jb)
                    i3, _ = intertwiner_range(jb, j25, jb, jb)

                    isempty(i1) && continue
                    isempty(i2) && continue
                    isempty(i3) && continue
                    isempty(i4) && continue

                    # must be computed
                    push!(spin_j45, j45)

                end

                isempty(spin_j45) && continue

                # must be computed
                push!(spin_j23, j23)
                push!(spin_j45_ext, spin_j45)

            end

            if (isempty(spin_j23) || isempty(spin_j45_ext))
                continue
            end

            # must be computed
            push!(spin_j25_j34_pcutoff, [j25, j34])
            push!(spin_j23_pcutoff, spin_j23)            
            push!(spin_j45_pcutoff, spin_j45_ext)

        end

        if isempty(spin_j25_j34_pcutoff)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spin_index_1 in eachindex(spin_j25_j34_pcutoff)

            j25 = spin_j25_j34_pcutoff[spin_index_1][1]
            j34 = spin_j25_j34_pcutoff[spin_index_1][2]

            # range of intertwiners
            i3, i3_range = intertwiner_range(jb, j25, jb, jb)
            i2, i2_range = intertwiner_range(j34, jb, j25, j34)

            # dim internal faces
            dfj = dim(j25) * dim(j34)

            ampt = zeros(number_of_threads)

            Threads.@threads for spin_index_2 in eachindex(spin_j23_pcutoff[spin_index_1])

                j23 = spin_j23_pcutoff[spin_index_1][spin_index_2]

                # range of intertwiners
                i4, i4_range = intertwiner_range(j34, j23, jb, jb)
                reduced_range_v2 = ((0, 0), i4, i2, i4, i3)

                # compute second EPRL vertex
                v2 = vertex_compute([jb, jb, jb, jb, j34, j23, jb, j34, j25, jb], shells, reduced_range_v2; result=result_return)

                amp_2 = 0.0

                for j45 in spin_j45_pcutoff[spin_index_1][spin_index_2]

                    # range of intertwiners
                    i1, i1_range = intertwiner_range(jb, jb, j45, j34)
                    reduced_range_v1 = ((0, 0), i3, i1, i2, i1)

                    # compute first EPRL vertex
                    v1 = vertex_compute([jb, jb, jb, jb, jb, j25, jb, j34, j45, j34], shells, reduced_range_v1; result=result_return)

                    amp_3 = 0.0

                    # LoopVectorization not safe
                    @inbounds for i1 in 1:i1_range, i2 in 1:i2_range, i3 in 1:i3_range, i4 in 1:i4_range
                        amp_3 += v1.a[i1, i2, i1, i3, 1] * v2.a[i3, i4, i2, i4, 1]
                    end

                    amp_2 += amp_3 * dim(j45)

                end

                ampt[Threads.threadid()] += amp_2 * dim(j23)

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
@time frogface_EPRL(1, 0);
println("done\n")
sleep(1)


ampls_matrix = Array{Float64,2}(undef, convert(Int, 2 * CUTOFF + 1), SHELL_MAX - SHELL_MIN + 1)

printstyled("\nStarting computation with K = $(CUTOFF), Dl_min = $(SHELL_MIN), Dl_max = $(SHELL_MAX), Immirzi = $(IMMIRZI)...\n"; bold=true, color=:cyan)

column_labels = String[]

for Dl = SHELL_MIN:SHELL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)
    @time ampls = frogface_EPRL(CUTOFF, Dl)
    push!(column_labels, "Dl = $(Dl)")
    ampls_matrix[:, Dl-SHELL_MIN+1] = ampls[:]

end

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(ampls_matrix, column_labels)
CSV.write("$(STORE_FOLDER)/frogface_2_Dl_min_$(SHELL_MIN)_Dl_max_$(SHELL_MAX).csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
