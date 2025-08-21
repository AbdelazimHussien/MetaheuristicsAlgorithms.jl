using MetaheuristicsAlgorithms
using Documenter

DocMeta.setdocmeta!(MetaheuristicsAlgorithms, :DocTestSetup, :(using MetaheuristicsAlgorithms); recursive=true)

makedocs(;
    modules=[MetaheuristicsAlgorithms],
    authors="Abdelazim Hussien",
    sitename="MetaheuristicsAlgorithms.jl",
    format=Documenter.HTML(size_threshold=300_000;
        canonical="https://abdelazimhussien.github.io/MetaheuristicsAlgorithms.jl",
        edit_link="main",
        assets=String[],
        prettyurls=false
        # prettyurls = get(ENV, "CI", "false") == "true"
    ),
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
        "List" => "AlgList.md",
        "Engineering Problems" => "constrainengineering.md",
        "CEC" => "cec.md",
        "Examples" => "example.md"
    ],
    # html = Documenter.HTMLWriter.HTML(size_threshold=300_000),
)

deploydocs(;
    # repo="github.com/abdelazimhussien/MetaheuristicsAlgorithms.jl",
    # devbranch="master",
    # repo="github.com/AbdelazimHussien/MetaheuristicsAlgorithms.jl.git",
    repo="https://github.com/AbdelazimHussien/MetaheuristicsAlgorithms.jl.git",
    devbranch="main",  
    branch="gh-pages"
)


# deploydocs(
#     repo="github.com/abdelazimhussien/MetaheuristicsAlgorithms.jl",
#     devbranch="master",
#     branch="gh-pages"  # Explicitly set the target branch
# )

