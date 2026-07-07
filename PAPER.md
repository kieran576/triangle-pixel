## Abstract

Real-time autonomous systems---drones, robots, autonomous vehicles---require instant visual perception, yet the traditional Bayer sensor pipeline introduces tens of milliseconds of ISP latency before any color image is available. We propose Triangle Pixel, a vision system built on equilateral triangle meshes that enables **zero-latency color perception**: each hexagonal group of 6 triangles naturally contains 2R+2G+2B, producing a perceptually full-color image directly from raw sensor data without any computation. For precision tasks, the same triangular raw data supports a full reconstruction pipeline. Critically, every triangle captures one ground-truth channel measurement at its position, unlike Bayer sensors where two-thirds of per-pixel color values are interpolated. This measurement fidelity benefits downstream computational photography. The zero-latency raw path and the full ISP pipeline both run at 59 FPS on a consumer CPU.

## 1. Introduction

Real-time autonomous systems---drones navigating cluttered environments, robots performing high-speed manipulation, autonomous vehicles executing emergency braking---require visual perception with minimal latency. The standard Bayer-filtered camera pipeline imposes a bottleneck: raw sensor data must pass through a demosaicing ISP (typically 10-50 ms on embedded hardware) before any color-based decision can be made. For applications where milliseconds determine safety outcomes, this latency is unacceptable.

Beyond latency, the Bayer pattern introduces a fundamental data fidelity problem. Each pixel measures only one color channel; the other two are *interpolated* from neighbors. Every subsequent computational step---HDR fusion, super-resolution, denoising, object detection---operates on data where two-thirds of the per-pixel color values are estimates, not measurements. Interpolation errors compound through the pipeline.

We propose a sensor topology addressing both problems: photodiodes on an equilateral triangular lattice with a 6-triangle repeating color filter assignment. This provides three key advantages:

1. **Zero-latency perception**: Each hexagonal group of 6 triangles around any vertex contains exactly 2R+2G+2B (Property 1). With no computation, the raw sensor output produces a perceptually full-color image---enabling instant color-based decision-making directly from the sensor, without an ISP.
2. **Measurement fidelity**: Every triangle captures one ground-truth channel value at its exact position. Property 2 guarantees three distinct neighbor channels. Computational photography pipelines operate on data with one verified measurement per element.
3. **Geometric structure**: The lattice has 6-fold rotational symmetry (vs. 4-fold), providing near-isotropic response (2.1 dB anisotropy). Sierpinski self-similarity enables natural multi-scale pyramids. Triangular faces map directly to 3D mesh surfaces.

Fujifilm X-Trans, Foveon X3, and learned demosaicing improve upon Bayer but retain rectangular grids---they cannot exploit hexagonal color balance for zero-latency perception.

We present a complete vision pipeline---from sensor simulation through zero-latency perception, image reconstruction, edge detection, feature matching, 3D mesh generation, to AI-native processing---all operating on the triangular mesh.

## 9. Conclusion

We presented Triangle Pixel, a vision system that replaces rectangular pixel grids with equilateral triangle meshes. Its defining capability is **zero-latency color perception**: the raw triangular sensor output is directly viewable as a full-color image without any ISP processing, because each hexagonal group naturally balances 2R+2G+2B. For precision tasks, the same raw data supports computational reconstruction, and every triangle contributes one ground-truth channel measurement---unlike Bayer sensors where per-pixel color values are predominantly interpolated. The pipeline operates at 59 FPS on a consumer CPU across both the zero-latency raw path and the full ISP path. This combination of instant perception, measurement fidelity, and geometric structure makes triangular sensing a candidate foundation for next-generation autonomous vision systems.


