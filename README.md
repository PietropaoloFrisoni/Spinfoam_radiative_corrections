# Radiative corrections of the spin foam propagator

***The codes are parallelized on the available cores.***

*It is advisable for the performance to use a number of workers equal to or less than the physical number of cores present on the system.*

A full list of the employed julia packages can be found in ./julia_codes/pkgs.jl.

*Before executing the codes, all packages must be installed.*

**Notice that the julia's Just-in-Time compiler is such that the first execution of functions is considerably slower that following ones, and it also allocates much more memory**. To avoid this, you can use the [DaemonMode package](https://github.com/dmolina/DaemonMode.jl).


## Usage:

in order to execute the codes (on a single machine with the synthax below) you can run the following command:

```
$JULIA_EXECUTABLE_PATH    -p    [N-1]    $JULIA_CODE_PATH    $ARGS
```

where [N-1] is the number of workers. For the non-parallel version just omit the "-p    [N-1]".


            




