import Pkg

Pkg.activate(".")
Pkg.instantiate()

packages = [
    "RandomNumbers",
]

for pkg in packages
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg" exception = e
    end
end

# The symbolics package can sometimes fail in this. You can just skip it as it is not
# needed to run the package
unregistered_packages = [
    "https://github.com/peterrrock2/Multi-Scale-Map-Sampler",
]

for pkg_url in unregistered_packages
    try
        Pkg.add(url=pkg_url)
    catch e
        @warn "Failed to install package from $pkg_url" exception = e
    end
end

println("Environment setup is complete.")

