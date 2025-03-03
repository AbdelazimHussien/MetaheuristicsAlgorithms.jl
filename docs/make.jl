using MetaheuristicsAlgorithms
using Documenter

DocMeta.setdocmeta!(
    MetaheuristicsAlgorithms,
    :DocTestSetup,
    :(using MetaheuristicsAlgorithms);
    recursive=true,
)

makedocs(;
    modules=[MetaheuristicsAlgorithms],
    authors="Abdelazim Hussien",
    sitename="MetaheuristicsAlgorithms.jl",
    format=Documenter.HTML(;
        canonical="https://abdelazimhussien.github.io/MetaheuristicsAlgorithms.jl",
        edit_link="main",
        assets=String[],
        prettyurls=false,
        # prettyurls = get(ENV, "CI", "false") == "true"
    ),
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
        "List" => "AlgList.md",
        "Examples" => "example.md",
    ],
)

deploydocs(
    repo="github.com/abdelazimhussien/MetaheuristicsAlgorithms.jl",
    devbranch="main",
    branch="gh-pages",  # Explicitly set the target branch
)
