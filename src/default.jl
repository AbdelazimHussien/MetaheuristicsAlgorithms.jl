include("AEFA.jl")
include("AEO.jl")
include("AFT.jl")
include("AHA.jl")
include("ALO.jl")
include("AOArithmetic.jl")
include("APO.jl")
include("ARO.jl")
include("AVOA.jl")
include("BES.jl")
include("BKA.jl")
include("BO.jl")
include("BOA.jl")
# include("CFOA.jl")
include("ChOA.jl")
include("CO.jl")
include("CoatiOA.jl")
include("COOT.jl")
include("CSBO.jl")
include("DBO.jl")
include("DDAO.jl")
include("DMOA.jl")
include("DO.jl")
include("DSO.jl")
include("ECO.jl")
include("EDO.jl")
include("ElkHO.jl")
include("EO.jl")
include("ESC.jl")
include("ETO.jl")
include("FLA.jl")
include("FLoodA.jl")
include("FOX.jl")
# include("FPA.jl")
include("GazelleOA.jl")
include("GBO.jl")
include("GEA.jl")
include("GJO.jl")
include("GKSO.jl")
include("GNDO.jl")
include("GO.jl")
include("GOA.jl")
include("GTO.jl")
include("GWO.jl")
include("HBA.jl")
include("HBO.jl")
include("HEOA.jl")
include("HGS.jl")
include("HGSO.jl")
include("HHO.jl")
include("HikingOA.jl")
include("HO.jl")
include("HorseOA.jl")

include("INFO.jl")
include("IVYA.jl")
include("Jaya.jl")
include("JS.jl")
include("LCA.jl")
include("LFD.jl")
include("LPO.jl")
include("MossGO.jl")
include("MountainGO.jl")
include("MPA.jl")
include("MRFO.jl")
include("MVO.jl")
include("OOA.jl")
include("ParrotO.jl")
include("PDO.jl")
include("PKO.jl")
include("PLO.jl")
include("POA.jl")
include("PoliticalO.jl")
include("PumaO.jl")
include("QIO.jl")
include("RBMO.jl")
include("ROA.jl")
include("RSA.jl")
include("RSO.jl")
include("RUN.jl")
include("SBO.jl")
include("SBOA.jl")
include("SCHO.jl")
include("SeaHO.jl")
include("SFOA.jl")
include("SHO.jl")
include("SMA.jl")
include("SnowOA.jl")
include("SO.jl")
include("SparrowSA.jl")
include("STOA.jl")
include("SupplyDO.jl")
include("TLBO.jl")
include("TLCO.jl")
include("TSA.jl")
include("TTAO.jl")
include("WHO.jl")
include("WO.jl")
include("WOA.jl")
include("WSO.jl")
include("YDSE.jl")
include("ZOA.jl")
##
include("Chung_Reynolds.jl")
include("initialization.jl")
##
dim = 30
Max_iteration = 1000
SearchAgents_no = 50
lb = -100
ub = 100
tlt = "Chung Reynolds"
i = 1
#
BestPosition, BestValue, ConvergenceCurve = ESC(SearchAgents_no, Max_iteration, lb, ub, dim, Chung_Reynolds)
println("BestValue: ", BestPosition)