# Radiative corrections to the EPRL spin foam propagator

***The julia codes are parallelized on the available cores using a hybrid multilevel parallelization scheme, exploiting the available processes, threads and loop vectorization.*** _It is therefore advisable for the performance to use a number of workers * threads equal to or less than the physical number of cores present on the system._

A full list of the employed julia packages can be found in ./julia_codes/pkgs.jl. _Before executing the codes, all packages must be installed._

**Notice that the julia's Just-in-Time compiler is such that the first execution of functions is considerably slower that following ones, and it also allocates much more memory**. To avoid this, you can use the [DaemonMode package](https://github.com/dmolina/DaemonMode.jl).

We provide the ipe file *diagrams_code_notation* with the explicit structure of the spinfoams corresponding to the diagrams. The spins and intertwiners labels match exactly the ones implemented in the julia scripts. _We provide the computed data in the *data* folder._

Finally, we leave the Mathematica notebooks used for the data analysis in the *Notebooks* folder.

## Usage:

in order to execute the julia codes (on a single machine with the synthax below) you can run the following command:

```
$JULIA_EXECUTABLE_PATH   -p   [N-1]   --threads   [T]   $JULIA_CODE_PATH   $ARGS
```

where [N-1] is the number of workers and [T] is the number of threads. **For an explicit example, see the *CC_script.sh* file**, which was used to run the codes on Compute Canada's Graham cluster. 
