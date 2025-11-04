module MetaheuristicsAlgorithms

using Random,
    Dates,
    SpecialFunctions,
    Distributions,
    Statistics,
    LinearAlgebra,
    Printf

# abstract type OptimizationResult end 

struct OptimizationResult
    bestX::Vector{Float64}
    bestF::Float64
    his_best_fit::Vector{Float64}
end
export OptimizationResult
include("problem.jl")

# using .ProblemDefinition
export OptimizationProblem 

include("plotting/convergence.jl")
export convergence_curve


include("core/optimize.jl")
export optimize


include("compare_algorithms.jl")
export compare_algorithms

include("benchmark/CEC2005.jl")
include("benchmark/enginnering_problems.jl")
export  Engineering_F1, Engineering_F2, Engineering_F3, Engineering_F4, Engineering_F5, Engineering_F6, Engineering_F7, Engineering_F8, Engineering_F9



include("algo/AEFA.jl")
include("algo/AEO.jl")
include("algo/AFT.jl")
include("algo/AHA.jl")
include("algo/ALA.jl")
include("algo/ALO.jl")
include("algo/AOArithmetic.jl")
include("algo/APO.jl")
include("algo/ARO.jl")
include("algo/ArtemisininO.jl")
include("algo/AVOA.jl")
include("algo/BES.jl")
include("algo/BKA.jl")
include("algo/BO.jl")
include("algo/BOA.jl")
include("algo/CDO.jl")
include("algo/ChameleonSA.jl")
include("algo/CapSA.jl")
include("algo/ChOA.jl")
include("algo/CO.jl")
include("algo/CoatiOA.jl")
include("algo/COOT.jl")
include("algo/CSBO.jl")
include("algo/DBO.jl")
include("algo/DDAO.jl")
include("algo/DMOA.jl")
include("algo/DO.jl")
include("algo/DSO.jl")
include("algo/ECO.jl")
include("algo/EDO.jl")
include("algo/ElkHO.jl")
include("algo/EO.jl")
include("algo/ETO.jl")
include("algo/ESC.jl")
include("algo/FATA.jl")
include("algo/FLA.jl")
include("algo/FLoodA.jl")
include("algo/FOX.jl")
include("algo/GazelleOA.jl")
include("algo/GBO.jl")
include("algo/GEA.jl")
include("algo/GGO.jl")
include("algo/GJO.jl")
include("algo/GKSO.jl")
include("algo/GNDO.jl")
include("algo/GO.jl")
include("algo/GOA.jl")
include("algo/GTO.jl")
include("algo/GWO.jl")
include("algo/HBA.jl")
include("algo/HBO.jl")
include("algo/HEOA.jl")
include("algo/HGS.jl")
include("algo/HGSO.jl")
include("algo/HHO.jl")
include("algo/HikingOA.jl")
include("algo/HO.jl")
include("algo/HorseOA.jl")
include("algo/INFO.jl")
include("algo/IVYA.jl")
include("algo/Jaya.jl")
include("algo/JS.jl")
include("algo/LCA.jl")
include("algo/LFD.jl")
include("algo/LPO.jl")
include("algo/LSHADE_cnEpSin.jl")
include("algo/MossGO.jl")
include("algo/MountainGO.jl")
include("algo/MPA.jl")
include("algo/MRFO.jl")
include("algo/MVO.jl")
include("algo/OOA.jl")
include("algo/ParrotO.jl")
include("algo/RBMO.jl")
include("algo/PDO.jl")
include("algo/PKO.jl")
include("algo/PLO.jl")
include("algo/POA.jl")
include("algo/PoliticalO.jl")
include("algo/PRO.jl")
include("algo/PumaO.jl")
include("algo/QIO.jl")
include("algo/ROA.jl")
include("algo/RFO.jl")
include("algo/RIME.jl")
include("algo/RSA.jl")
include("algo/RSO.jl")
include("algo/RUN.jl")
include("algo/SBO.jl")
include("algo/SBOA.jl")
include("algo/SCA.jl")
include("algo/SCHO.jl")
include("algo/SeaHO.jl")
include("algo/SFOA.jl")
include("algo/SHO.jl")
include("algo/SMA.jl")
include("algo/SnowOA.jl")
include("algo/SO.jl")
include("algo/SOA.jl")
include("algo/SparrowSA.jl")
include("algo/SSA.jl")
include("algo/STOA.jl")
include("algo/SuperbFOA.jl")
include("algo/SupplyDO.jl")
include("algo/THRO.jl")
include("algo/TLBO.jl")
include("algo/TLCO.jl")
include("algo/TOC.jl")
include("algo/TSA.jl")
include("algo/TTAO.jl")
include("algo/WHO.jl")
include("algo/WO.jl")
include("algo/WOA.jl")
include("algo/WSO.jl")
include("algo/WUTP.jl")
include("algo/YDSE.jl")
include("algo/ZOA.jl")

using SpecialFunctions: gamma

include("utils/utils.jl")
include("utils/Chung_Reynolds.jl")
include("utils/initialization.jl")
# addpath(joinpath(@__DIR__, "algorithms"))

##
# using .AEFA, .AEO, .AFT, .AHA, .ALO, .AOArithmetic, .APO, .ARO, .ArtemisininO, .AVOA
# using .AEO, .AFT, .AHA, .ALO, .AOArithmetic, .APO, .ARO, .ArtemisininO, .AVOA
# using .BES, .BKA, .BO, .BOA
# using .CDO, .Chameleon, .ChOA, .CO, .CoatiOA, .COOT, .CSBO
# using .DBO, .DDAO, .DMOA, .DO, .DSO
# using .ECO, .EDO, .ElkHO, .EO, .ETO
# using .FATA, .FLA, .FLoodA, .FOX
# using .GazelleOA, .GBO, .GEA, .GGO, .GJO, .GKSO, .GNDO, .GO, .GOA, .GTO, .GWO
# using .HBA, .HBO, .HEOA, .HGS, .HGSO, .HikingOA, .HO, .HorseOA
# using .INFO, .IVYA
# using .Jaya, .JS
# using .LCA, .LFD, .LPO
# using .MossGO, .MountainGO, .MPA, .MRFO, .MVO
# using .OOA
# using .ParrotO, .PDO, .PKO, .PLO, .POA, .PoliticalO, .PRO, .PumaO
# using .QIO
# using .ROA, .RSA, .RSO, .RUN
# using .SBO, .SBOA, .SCA, .SCHO, .SeaHO, .SHO, .SMA, .SnowOA, .SO, .SparrowSA, .SSA, .STOA, .SupplyDO
# using .TLBO, .TLCO, .TSA, .TTAO
# using .WHO, .WO, .WOA, .WSO
# using .YDSE
# using .ZOA
# This list of exports include 
# repeated items and should be cleaned up.
# some of the algorithms are missing and didn't get exported.
export AEFA, AEO, AFT, AHA, ALA, ALO, AOArithmetic, APO, ARO, AVOA
export BES, BKA, BO, BOA
export CapSA, ChOA, CO, CoatiOA, COOT, CSBO
export DBO, DDAO, DMOA, DO, DSO
export ECO, EDO, ElkHO, EO, ETO, ESC
export FLA, FLoodA, FOX
export GazelleOA, GBO, GEA, GGO, GJO, GKSO, GNDO, GO, GOA, GTO, GWO
export HBA, HBO, HEOA, HGS, HGSO, HikingOA, HO, HorseOA
export INFO, IVYA
export Jaya, JS
export LCA, LFD, LPO, LSHADE_cnEpSin
# export LSHADE_cnEpSin
export RBMO, RFO, RIME, ROA, RSA, RSO, RUN
export SFOA, SHO, SMA, SSA, STOA, SupplyDO
export THRO, TLBO, TLCO, TSA, TTAO, WHO, WO, WOA, WSO, YDSE, ZOA

end
