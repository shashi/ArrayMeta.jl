module ArrayMeta

using MacroTools
using Base.Test

include("util.jl")
include("lowering.jl")
include("impl/abstract.jl")
include("impl/dagger.jl")

end # module
