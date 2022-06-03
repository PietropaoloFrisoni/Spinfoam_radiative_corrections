using Distributed

printstyled("\nSelf-energy BF divergence parallelized on $(nworkers()) worker(s)\n\n"; bold=true, color=:blue)

length(ARGS) < 3 && error("please use these 3 arguments: data_sl2cfoam_next_folder    cutoff    store_folder")
@eval @everywhere DATA_SL2CFOAM_FOLDER = $(ARGS[1])
CUTOFF = parse(Int, ARGS[2])
@eval STORE_FOLDER = $(ARGS[3])

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

STORE_FOLDER = "$(STORE_FOLDER)/data/BF/cutoff_$(CUTOFF_FLOAT)"
mkpath(STORE_FOLDER)

printstyled("initializing library...\n"; bold=true, color=:cyan)
@everywhere init_sl2cfoam_next(DATA_SL2CFOAM_FOLDER, 0.123) # fictitious Immirzi 
println("done\n")


function self_energy(cutoff)

    number_of_threads = Threads.nthreads()

    # set boundary
    step = onehalf = half(1)
    jb = half(1)

    ampls = Float64[]
    
    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff
        
        # generate a list of all spins to compute
        spins_all = NTuple{2, HalfInt}[]
        for j23::HalfInt = 0:onehalf:pcutoff, j25::HalfInt = 0:onehalf:pcutoff
            
            # skip if computed in lower partial cutoff
            j23 <= (pcutoff-step) && j25 <= (pcutoff-step) && continue

            # skip if any intertwiner range empty
            i2, _ = intertwiner_range(jb, j25, jb, j23)
            i3, _ = intertwiner_range(j23, jb, jb, j23)

            isempty(i2) && continue
            isempty(i3) && continue

            # must be computed
            push!(spins_all, (j23, j25))

        end

        if isempty(spins_all)
            push!(ampls, 0.0)
            continue
        end

        @time tampl = @sync @distributed (+) for spins in spins_all

            j23, j25 = spins

            # restricted range of intertwiners
            i2, i2_range = intertwiner_range(jb, j25, jb, j23)
            i3, i3_range = intertwiner_range(j23, jb, jb, j23)
            i4, i4_range = intertwiner_range(j23, jb, jb, j23)
            i5, i5_range = intertwiner_range(jb, j25, jb, j23)

            # compute vertex
            v = vertex_BF_compute([jb, jb, jb, jb, j23, jb, j25, j23, jb, j23])

            # contract
            dfj = dim(j23) * dim(j25)
            
            # intertwiner contractions
            amp = 0.0

            for i2 in 1:i2_range, i3 in 1:i3_range, i4 in 1:i4_range, i5 in 1:i5_range
                amp += v.a[i2, i4, i3, i5, 1] * v.a[i5, i4, i3, i2, 1]
            end

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
@time self_energy(1);
println("done\n")
sleep(1)

printstyled("\nStarting computation with K = $(CUTOFF)...\n"; bold=true, color=:cyan)
@time ampls = self_energy(CUTOFF);

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(amplitudes=ampls)
CSV.write("$(STORE_FOLDER)/self_energy.csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)


