# Radiative corrections to the EPRL spin foam propagator

**_The julia codes are parallelized on the available cores using a hybrid multilevel parallelization scheme, exploiting the available processes, threads and loop vectorization._** _It is therefore advisable for the performance to use a number of workers \* threads equal to or less than the physical number of cores present on the system._

A full list of the employed julia packages can be found in `./julia*codes/pkgs.jl`. \_Before executing the codes, all packages must be installed.\*

**Notice that the julia's Just-in-Time compiler is such that the first execution of functions is considerably slower that following ones, and it also allocates much more memory**. To avoid this, you can use the [DaemonMode package](https://github.com/dmolina/DaemonMode.jl).

The [ipe](https://ipe.otfried.org/) file `diagrams_code_notation.ipe` contains the explicit structure of the spinfoams diagrams. The spins and intertwiners labels match exactly the ones implemented in the julia scripts. We provide the computed data in the `data` folder.

We provide the Mathematica notebooks used for the data analysis in the `Notebooks` folder.

## Usage:

in order to execute the julia codes (on a single machine with the synthax below) you can run the following command:

```
$JULIA_EXECUTABLE_PATH   -p   [N-1]   --threads   [T]   $JULIA_CODE_PATH   $ARGS
```

where [N-1] is the number of workers and [T] is the number of threads. **For an explicit example, see the _CC_script.sh_ file**, which was used to run the codes on Compute Canada's Graham cluster.
