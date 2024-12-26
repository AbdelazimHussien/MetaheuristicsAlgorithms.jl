using MetaheuristicsAlgorithms
using Documenter

DocMeta.setdocmeta!(MetaheuristicsAlgorithms, :DocTestSetup, :(using MetaheuristicsAlgorithms); recursive=true)

makedocs(;
    modules=[MetaheuristicsAlgorithms],
    authors="Abdelazim Hussien",
    sitename="MetaheuristicsAlgorithms.jl",
    format=Documenter.HTML(;
        canonical="https://abdelazimhussien.github.io/MetaheuristicsAlgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/abdelazimhussien/MetaheuristicsAlgorithms.jl",
    devbranch="main",
)
