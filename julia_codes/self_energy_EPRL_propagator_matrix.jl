using Distributed

number_of_workers = nworkers()
number_of_processes = nprocs()
number_of_threads = Threads.nthreads()
available_cpus = Sys.cpu_info()

printstyled("\nSelf energy EPRL propagator matrix parallelized on $(number_of_workers) worker(s) and $(number_of_threads) thread(s)\n\n"; bold=true, color=:blue)

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

function self_energy_EPRL_propagator_matrix(cutoff, shells, jb)

    step = onehalf = half(1)

    DIM = dim(jb)
    propagator_matrix = zeros(DIM, DIM, convert(Int, 2 * cutoff + 1))
    @everywhere accumulated_propagator_matrix = zeros($DIM, $DIM)

    result_return = (ret=true, store=false, store_batches=false)

    pcutoff_index = 0

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a list of bulk spins to compute
        bulk_spins_pcutoff = NTuple{2,HalfInt}[]

        # generate a list of all spins to compute
        bulk_spins_pcutoff = NTuple{2,HalfInt}[]
        for j23::HalfInt = 0:onehalf:pcutoff, j25::HalfInt = 0:onehalf:pcutoff

            # skip if computed in lower partial cutoff
            j23 <= (pcutoff - step) && j25 <= (pcutoff - step) && continue

            # skip if any intertwiner range empty
            i2, _ = intertwiner_range(jb, j25, jb, j23)
            i3, _ = intertwiner_range(j23, jb, jb, j23)

            isempty(i2) && continue
            isempty(i3) && continue

            # must be computed
            push!(bulk_spins_pcutoff, (j23, j25))

        end

        if isempty(bulk_spins_pcutoff)
            pcutoff_index += 1
            for column_index = 1:1:DIM, row_index = 1:1:DIM
                propagator_matrix[row_index, column_index, pcutoff_index] = 0.0
            end
            continue
        end

        @sync @distributed for bulk_spins in bulk_spins_pcutoff

            j23, j25 = bulk_spins

            # restricted range of intertwiners
            i2, i2_range = intertwiner_range(jb, j25, jb, j23)
            i3, i3_range = intertwiner_range(j23, jb, jb, j23)
            i4, i4_range = intertwiner_range(j23, jb, jb, j23)
            i5, i5_range = intertwiner_range(jb, j25, jb, j23)

            # compute vertex
            v = vertex_compute([jb, jb, jb, jb, j23, jb, j25, j23, jb, j23], shells; result=result_return)

            # contract
            dfj = dim(j23) * dim(j25)

            for column_index = 1:1:DIM, row_index = column_index:1:DIM

                # intertwiner contractions
                amp = 0.0

                for i2 in 1:i2_range, i3 in 1:i3_range, i4 in 1:i4_range, i5 in 1:i5_range
                    amp += v.a[i2, i4, i3, i5, row_index] * v.a[i5, i4, i3, i2, column_index]
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

        println("Matrix propagator at cutoff $pcutoff is equal to:")
        println(propagator_matrix[:, :, pcutoff_index], "\n")

    end # partial cutoffs loop

    propagator_matrix

end

printstyled("Pre-compiling the function...\n"; bold=true, color=:cyan)
@time self_energy_EPRL_propagator_matrix(1, 0, half(1));
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
    @time propagator_matrix = self_energy_EPRL_propagator_matrix(CUTOFF, Dl, BOUNDARY_SPIN)

    printstyled("\nSaving dataframes...\n"; bold=true, color=:cyan)
    FINAL_SHELL_PATH = "$(STORE_FOLDER)/Dl_$(Dl)"
    mkpath(FINAL_SHELL_PATH)

    for pcutoff = 0:0.5:CUTOFF_FLOAT
        df = DataFrame(propagator_matrix[:, :, convert(Int, 2 * pcutoff + 1)], column_labels)
        CSV.write("$(FINAL_SHELL_PATH)/self_energy_pcutoff_$(pcutoff).csv", df)
    end

end

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
