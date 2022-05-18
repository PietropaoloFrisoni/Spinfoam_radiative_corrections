using Distributed

printstyled("\nFrogface BF divergence parallelized on $(nworkers()) worker(s)\n\n"; bold=true, color=:blue)

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

function frogface_BF(cutoff)

    # set boundary
    step = onehalf = half(1)
    jb = half(1)
    # ib must be in range [0, 2jb]
    # (julia index starts from 1)
    ib_index = 1

    ampls = Float64[]

    # loop over partial cutoffs
    for pcutoff = 0:step:cutoff

        # generate a list of bulk spins to compute
        bulk_spins_pcutoff = NTuple{4,HalfInt}[]

        for j25::HalfInt = 0:onehalf:pcutoff, j34::HalfInt = 0:onehalf:pcutoff,
            j23::HalfInt = 0:onehalf:pcutoff, j45::HalfInt = 0:onehalf:pcutoff

            # skip if computed in lower partial cutoff
            j25 <= (pcutoff - step) && j34 <= (pcutoff - step) &&
            j23 <= (pcutoff - step) && j45 <= (pcutoff - step) && continue

            # skip if any intertwiner range empty
            i2, _ = intertwiner_range(j23, jb, j25, jb)
            i3, _ = intertwiner_range(j34, jb, jb, jb)
            i4, _ = intertwiner_range(jb, j25, j25, j34)
            i5, _ = intertwiner_range(j45, jb, j25, jb)

            isempty(i2) && continue
            isempty(i3) && continue
            isempty(i4) && continue
            isempty(i5) && continue

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
            i2, i2_range = intertwiner_range(j23, jb, j25, jb)
            i3, i3_range = intertwiner_range(j34, jb, jb, jb)
            i4, i4_range = intertwiner_range(jb, j25, j25, j34)
            i5, i5_range = intertwiner_range(j45, jb, j25, jb)

            # compute BF vertices
            v1 = vertex_BF_compute([jb, jb, jb, jb, j34, j25, j25, jb, jb, j45])
            v2 = vertex_BF_compute([jb, jb, jb, jb, j23, jb, j25, jb, j25, j34])

            # dim internal faces
            dfj = dim(j25) * dim(j34) * dim(j45) * dim(j23)

            # intertwiner contractions
            amp = 0.0
            
            for i2 in 1:i2_range, i3 in 1:i3_range, i4 in 1:i4_range, i5 in 1:i5_range
                amp += v1.a[i5, i5, i3, i4, ib_index] * v2.a[i4, i3, i2, i2, ib_index]
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
@time frogface_BF(1);
println("done\n")
sleep(1)

printstyled("\nStarting computation with K = $(CUTOFF)...\n"; bold=true, color=:cyan)
@time ampls = frogface_BF(CUTOFF);

printstyled("\nSaving dataframe...\n"; bold=true, color=:cyan)
df = DataFrame(amplitudes=ampls)
CSV.write("$(STORE_FOLDER)/frogface.csv", df)

printstyled("\nCompleted\n\n"; bold=true, color=:blue)
