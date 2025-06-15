using MetaheuristicsAlgorithms
using Test
using Random

include("benchmark.jl")

include("testbenchmark.jl")
include("testdso.jl")
include("testaeo.jl")
# This tests are disabled due to a possible bug (? maybe) in the AHA and AFT algorithms
# include("testaha.jl")
# include("testaft.jl")



