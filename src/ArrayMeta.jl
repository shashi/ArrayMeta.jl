module ArrayMeta

using MacroTools

include("util.jl")
include("lowering.jl")
include("impl/abstract.jl")
include("impl/dagger.jl")

end # module
