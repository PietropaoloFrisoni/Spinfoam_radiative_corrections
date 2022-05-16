using Distributed

number_of_workers = nworkers()
number_of_processes = nprocs()
number_of_threads = Threads.nthreads()
available_cpus = Sys.cpu_info()

printstyled("\nHeptapod EPRL propagator matrix parallelized on $(number_of_workers) worker(s) and $(number_of_threads) thread(s)\n\n"; bold=true, color=:blue)

if (number_of_workers * number_of_threads > length(available_cpus))
    printstyled("WARNING: you are using more resources than available cores on this system. Performances will be affected\n\n"; bold=true, color=:red)
end

length(ARGS) < 7 && error("please use these 7 arguments: data_sl2cfoam_next_folder  cutoff  shell_min  shell_max   Immirzi  jboundary  store_folder")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
SHELL_MIN = parse(Int, ARGS[3])
SHELL_MAX = parse(Int, ARGS[4])
@eval @everywhere IMMIRZI = parse(Float64, $(ARGS[5]))
@eval STORE_FOLDER = $(ARGS[7])

printstyled("precompiling packages...\n"; bold=true, color=:cyan)
@everywhere begin
    include("pkgs.jl")
    include("init.jl")
end
println("done\n")

CUTOFF_FLOAT = parse(Float64, ARGS[2])
CUTOFF = HalfInt(CUTOFF_FLOAT)
BOUNDARY_SPIN_FLOAT = parse(Float64, ARGS[6])
BOUNDARY_SPIN = HalfInt(BOUNDARY_SPIN_FLOAT)

if (CUTOFF <= 1)
    error("please provide a larger cutoff")
end

STORE_FOLDER = "$(STORE_FOLDER)/data/EPRL/immirzi_$(IMMIRZI)/propagator_matrix/bspin_$(BOUNDARY_SPIN_FLOAT)/cutoff_$(CUTOFF_FLOAT)"
mkpath(STORE_FOLDER)

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, IMMIRZI)
println("done\n")

function heptapod_EPRL_propagator_matrix(cutoff, shells, jb)

    step = onehalf = half(1)

    DIM = dim(jb)
    propagator_matrix = zeros(DIM, DIM, convert(Int, 2 * cutoff + 1))
    @everywhere accumulated_propagator_matrix = zeros($DIM, $DIM)

    result_return = (ret=true, store=false, store_batches=false)

    pcutoff_index = 0

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff
        # generate a list of bulk spins to compute
        bulk_spins_pcutoff = NTuple{4,HalfInt}[]

        for j45::HalfInt = 0:onehalf:pcutoff, j34::HalfInt = 0:onehalf:pcutoff,
            j35::HalfInt = 0:onehalf:pcutoff, j78::HalfInt = 0:onehalf:pcutoff

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
            push!(bulk_spins_pcutoff, (j45, j34, j35, j78))

        end

        if isempty(bulk_spins_pcutoff)
            pcutoff_index += 1
            for column_index = 1:1:DIM, row_index = 1:1:DIM
                propagator_matrix[row_index, column_index, pcutoff_index] = 0.0
            end
            continue
        end

        @sync @distributed for bulk_spins in bulk_spins_pcutoff

            j45, j34, j35, j78 = bulk_spins

            # range of intertwiners
            i7, i7_range = intertwiner_range(jb, jb, jb, j78)
            i5, i5_range = intertwiner_range(j45, j35, jb, jb)
            i4, i4_range = intertwiner_range(j34, jb, jb, j45)
            i3, i3_range = intertwiner_range(jb, jb, j35, j34)

            # compute EPRL vertices
            v1 = vertex_compute([j45, jb, jb, j34, jb, jb, j35, jb, jb, jb], shells; result=result_return)
            v2 = vertex_compute([j34, jb, jb, j45, jb, jb, j35, j78, jb, jb], shells; result=result_return)

            # dim internal faces
            dfj = dim(j45) * dim(j34) * dim(j35) * dim(j78)

            for column_index = 1:1:DIM, row_index = column_index:1:DIM

                # intertwiner contractions
                amp = 0.0

                @inbounds for i7 in 1:i7_range, i3 in 1:i3_range, i4 in 1:i4_range, i5 in 1:i5_range
                    amp += v1.a[i3, row_index, column_index, i5, i4] * v2.a[i5, i7, i7, i3, i4]
                end

                accumulated_propagator_matrix[row_index, column_index] += amp * dfj

            end

        end

        CurrentMatrix = zeros(DIM, DIM)

        for p = 1:1:number_of_processes
            tmp_matrix = @retrieve_from_process(p, accumulated_propagator_matrix)
            for column_index = 1:1:DIM, row_index = column_index:1:DIM
                CurrentMatrix[row_index, column_index] += tmp_matrix[row_index, column_index]
            end
        end

        pcutoff_index += 1

        for column_index = 1:1:DIM, row_index = column_index:1:DIM
            propagator_matrix[row_index, column_index, pcutoff_index] = CurrentMatrix[row_index, column_index]
        end

        # symmetrize propagator matrix
        for row_index = 1:1:DIM, column_index = (row_index+1):1:DIM
            propagator_matrix[row_index, column_index, pcutoff_index] = propagator_matrix[column_index, row_index, pcutoff_index]
        end

        #println("Matrix propagator at cutoff $pcutoff is equal to:")
        #println(propagator_matrix[:, :, pcutoff_index], "\n")

    end # partial cutoffs loop

    propagator_matrix

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time heptapod_EPRL_propagator_matrix(1, 0, half(1));
println("done\n")
sleep(1)

column_labels = String[]
DIM = dim(BOUNDARY_SPIN)
for i = 1:DIM
    push!(column_labels, "i=$(i-1)")
end

printstyled("\nStarting computation with K = $(CUTOFF), Dl_min = $(SHELL_MIN), Dl_max = $(SHELL_MAX), Immirzi = $(IMMIRZI)...\n"; bold=true, color=:cyan)

for Dl = SHELL_MIN:SHELL_MAX

    printstyled("\nCurrent Dl = $(Dl)...\n"; bold=true, color=:magenta)
    @time propagator_matrix = heptapod_EPRL_propagator_matrix(CUTOFF, Dl, BOUNDARY_SPIN)

    printstyled("\nSaving dataframes...\n"; bold=true, color=:cyan)
    FINAL_SHELL_PATH = "$(STORE_FOLDER)/Dl_$(Dl)"
    mkpath(FINAL_SHELL_PATH)

    for pcutoff = 0:0.5:CUTOFF_FLOAT
        df = DataFrame(propagator_matrix[:, :, convert(Int, 2 * pcutoff + 1)], column_labels)
        CSV.write("$(FINAL_SHELL_PATH)/heptapod_pcutoff_$(pcutoff).csv", df)
    end

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
