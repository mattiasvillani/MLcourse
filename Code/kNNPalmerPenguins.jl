# k-NN analysis of the Palmer Penguin data

# Loading packages and importing functions
using CSV, DataFrames, Plots, LaTeXStrings, Measures
using MLJ, NearestNeighborModels, CategoricalArrays
utilFolder = "/home/mv/Dropbox/CurrentCode/UtilJulia/"
include(utilFolder*"MLUtil.jl") # for PlotClassifier2D()

# Plotting options
import ColorSchemes: Paired_12; 
colors = Paired_12[[1,2,7,8,3,4,5,6,9,10,11,12]]
Plots.reset_defaults()
gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8, 
    markersize = 4, markerstrokecolor = :auto)

# Load Palmer penguin data
penguins = DataFrame(CSV.File("/home/mv/Dropbox/Teaching/MLcourse/Data/PalmerPenguins.csv"))
rename!(penguins, :body_mass_kg => :bodymass, :flipper_length_cm => :flipperlength)
penguins.species = CategoricalArray(penguins.species)
Adelie = penguins[penguins.species .== "Adelie",:]
Gentoo = penguins[penguins.species .== "Gentoo",:]
Chinstrap = penguins[penguins.species .== "Chinstrap",:]

# Plot data
scatter(Adelie[!,:bodymass], Adelie[!,:flipperlength], label = "Adelie", 
    color = colors[1], xlabel = "body mass", ylabel ="flipper length")
scatter!(Gentoo[!,:bodymass], Gentoo[!,:flipperlength], 
    label = "Gentoo", color = colors[2])
scatter!(Chinstrap[!,:bodymass], Chinstrap[!,:flipperlength], 
    label = "Chinstrap", color = colors[4])
savefig("PalmerPenguinsData.svg")

# Fit kNN using MLJ and NearestNeighborModels packages
y, X = unpack(penguins, ==(:species), colname -> true);
kNN = @load KNNClassifier pkg = NearestNeighborModels
kNNmodel = kNN(K = 4)
mach = machine(kNNmodel, X, y)
fit!(mach)

# Plot the decision boundaries and overlay training data
predictFunc(x) = mode.(predict(mach, [x[1] x[2]]))[1]
PlotClassifier2D(y, X, predictFunc; gridSize = [300,300], 
    colors = colors[[1,2,9,10,3,4]], axisLabels = ["body mass","flipper length"])
        

