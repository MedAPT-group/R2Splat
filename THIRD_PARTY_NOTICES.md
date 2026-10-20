# Third-party notices

## gsplat

Parts of the CUDA implementation retain or adapt [gsplat](https://github.com/nerfstudio-project/gsplat)
material; others follow its design patterns. gsplat is developed by the
Nerfstudio project and its contributors.

The fixed reference snapshot is commit
[`e6d53183786d866af39c6baedb06db2ac0144517`][gsplat-commit].
Its [Apache-2.0 license][gsplat-license] is included unchanged in
[third_party/gsplat/LICENSE](third_party/gsplat/LICENSE).
License SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.

The [upstream tree][gsplat-tree] contains no `NOTICE` file; this excludes
submodule contents. The upstream GLM submodule is not vendored in R2Splat.

### Source map and local changes

Local paths below are under `r2splat/cuda/`; upstream paths are under
`gsplat/cuda/` at the fixed commit. Filenames after the first in each group
share its directory.

| Local file(s) | Upstream reference | Evidence and local changes |
| --- | --- | --- |
| `include/helpers.cuh` | [include/helpers.cuh][gsplat-helpers] | Inherited; complete file matches after CRLF normalization. Warp reductions and the `gsplat` namespace are retained. |
| `include/common.h` | [include/bindings.h][gsplat-bindings], [include/types.cuh][gsplat-types] | Macro block matches after prefix/whitespace normalization; `OpType` statements are retained. Local changes rename macros, split the header and guard CUDA scalar types. |
| `include/types.cuh` | [include/types.cuh][gsplat-types] | Source-declared design reference. Local vector/matrix structs replace GLM aliases and provide compatibility with the retained helpers. |
| `csrc/RasterizeToPixelsFwd.cu`, `RasterizeToPixelsBwd.cu` | [csrc/rasterize_to_pixels_fwd.cu][gsplat-raster-fwd], [csrc/rasterize_to_pixels_bwd.cu][gsplat-raster-bwd] | Parts of batch loads and launch setup match after whitespace normalization. Local forward/backward paths replace RGB/alpha compositing with additive CT integration, ray-length factors, Beta kernels and uncertainty. |
| `csrc/RasterizeToPixels.cpp`, `RasterizeToPixels.h` | [include/bindings.h][gsplat-bindings] and the rasterization files above | Tensor-wrapper/interface design reference; local CT arguments, validation, outputs and kernel dispatch. |
| `csrc/IsectTiles.cpp`, `IsectTiles.h`, `IsectTilesKernel.cu` | [csrc/isect_tiles.cu][gsplat-isect], [include/bindings.h][gsplat-bindings] | Source-declared design reference; separate counting/record generation, remove packed mode/depth ordering and add rectangular CT support. |
| `csrc/WorldToImageProjectionFwd.cu`, `WorldToImageProjectionBwd.cu` | [include/proj.cuh][gsplat-proj], [csrc/fully_fused_projection_fwd.cu][gsplat-projection-fwd], [csrc/fully_fused_projection_bwd.cu][gsplat-projection-bwd] | Source-declared perspective-clamping/gradient design references; local CT 3D projected covariance and ray-length axis, with separate forward/backward kernels. |
| `csrc/IsectTiles3D.cpp`, `IsectTiles3D.h`, `IsectTiles3DKernel.cu` | [csrc/isect_tiles.cu][gsplat-isect] (2D) | Design reference for a local 3D extension: split kernels, voxel-grid intersections and depth-independent sorting. No upstream `IsectTiles3D` file exists in this snapshot. |
| `csrc/VoxelizeToVoxels.cpp`, `VoxelizeToVoxels.h`, `VoxelizeToVoxelsFwd.cu`, `VoxelizeToVoxelsBwd.cu` | [2D rasterization forward][gsplat-raster-fwd]/[backward][gsplat-raster-bwd], [include/bindings.h][gsplat-bindings] | Design references for local 3D sampling, additive density, Beta kernels and uncertainty; backward uses retained `gsplat::warpSum`. No upstream `VoxelizeToVoxels` file exists in this snapshot. |

The original import revision cannot be established for every adapted file.
Design references indicate related upstream patterns, not verified historical
copies. Matching mathematical formulas alone do not establish code provenance.

## Runtime and build dependencies

PyTorch, the CUDA toolkit and compiler/build tools are external prerequisites
with their own licenses; their source code is not vendored here.

[gsplat-commit]: https://github.com/nerfstudio-project/gsplat/commit/e6d53183786d866af39c6baedb06db2ac0144517
[gsplat-license]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/LICENSE
[gsplat-tree]: https://api.github.com/repos/nerfstudio-project/gsplat/git/trees/e6d53183786d866af39c6baedb06db2ac0144517?recursive=1
[gsplat-helpers]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/include/helpers.cuh
[gsplat-bindings]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/include/bindings.h
[gsplat-types]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/include/types.cuh
[gsplat-raster-fwd]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu
[gsplat-raster-bwd]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/csrc/rasterize_to_pixels_bwd.cu
[gsplat-isect]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/csrc/isect_tiles.cu
[gsplat-proj]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/include/proj.cuh
[gsplat-projection-fwd]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/csrc/fully_fused_projection_fwd.cu
[gsplat-projection-bwd]: https://github.com/nerfstudio-project/gsplat/blob/e6d53183786d866af39c6baedb06db2ac0144517/gsplat/cuda/csrc/fully_fused_projection_bwd.cu
