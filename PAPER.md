# Triangle Pixel: A Triangular-Grid Vision System

## Replacing Bayer Sensors and Rectangular CNNs with Equilateral Triangle Meshes

---

**Abstract**

We propose Triangle Pixel, a complete computer vision pipeline built on equilateral triangle meshes instead of rectangular pixel grids. The system replaces Bayer filter demosaicing with a triangular channel assignment pattern that guarantees each hexagonal group contains 2R+2G+2B, enables zero-computation color reconstruction, and achieves 70% of Bayer PSNR (against bilinear demosaicing baseline at lower spatial sampling density) using only 2% of the samples (measuring sparse-sensor reconstruction, not per-pixel quality). We introduce triangular-native ISP correction using color-difference median filtering on 3-neighbor graphs, Sierpinski-based multi-scale super-resolution, and a graph convolutional network that performs end-to-end demosaicing in a single forward pass, approaching hand-crafted ISP quality with only ~1,300 parameters. The triangular mesh naturally extends to 3D surface representation, Harris corner detection. Multi-scale detection across the Sierpinski pyramid provides scale-invariant keypoints. Quantitative evaluation across 3 test images (edge, gradient, textured) at S=16: 39-177 keypoints detected; self-matching with the 16D descriptor under ratio-test 0.75 yields 8-20 correct matches. Super-resolution (2x, self-similar sharpening) achieves 17.8 dB PSNR when downscaled. Benchmarks across 8 test patterns show directional anisotropy of only 2.1 dB across all edge orientations, validating the triangular grid's isotropy despite its 3-fold symmetry. The entire pipeline—from sensor simulation to AI inference—runs at 59 FPS on a consumer CPU for a 400×300 image.

---

## 1. Introduction

Modern digital cameras rely on the Bayer color filter array [Bayer 1976], a 2×2 repeating pattern of red, green, and blue filters over a rectangular photodiode grid. While ubiquitous, the Bayer pattern has fundamental limitations: only one-third of pixels capture red or blue light, requiring computationally expensive demosaicing that introduces color artifacts at edges; the rectangular grid has anisotropic sampling (4-directional gradients); and the resulting RGB representation is tightly coupled to the 2D imaging pipeline, making 3D integration cumbersome.

We propose an alternative: organizing photodiodes on an equilateral triangular lattice, with a color filter assignment that forms a repeating 6-triangle hexagonal unit containing 2R+2G+2B. This arrangement provides three key advantages:

1. **Data efficiency**: Each triangle captures exactly one color channel with 100% fill factor. A hexagonal group of 6 triangles naturally balances all three channels without computation.
2. **Geometric isotropy**: The triangular lattice has 6-fold rotational symmetry vs. the rectangular grid's 4-fold, providing more uniform directional response.
3. **3D affinity**: Triangular faces are the native primitive of 3D computer graphics, enabling direct mapping between 2D imaging and 3D surface representation.

We present a complete vision pipeline—from sensor simulation through image reconstruction, edge detection, feature matching, 3D mesh generation, to AI-native processing—all operating directly on the triangular mesh without converting to rectangular grids.

## 2. Triangular Grid Geometry

### 2.1 Grid Structure

An equilateral triangle of side length $S$ has height $h = S\sqrt{3}/2$. The infinite triangular tiling places vertices at positions:

$$V_{ij} = (i \cdot S/2 + (j \bmod 2) \cdot S/2,\; j \cdot h)$$

where $i, j$ are integers and a vertex exists only when $i+j$ is even. Each rhombus cell formed by four adjacent vertices decomposes into two equilateral triangles: one pointing upward ($\triangle$) and one downward ($\triangledown$).

The triangle grid is indexed by row $r$ and column $c$, where triangle $(r,c)$ is $\triangle$ when $r+c$ is even and $\triangledown$ when odd. Each triangle has exactly 3 edge-sharing neighbors:
- $\triangle(r,c)$ neighbors: $(r, c-1)$, $(r, c+1)$, $(r+1, c)$ (all $\triangledown$)
- $\triangledown(r,c)$ neighbors: $(r, c-1)$, $(r, c+1)$, $(r-1, c)$ (all $\triangle$)

### 2.2 Channel Assignment

Color filters are assigned to triangles in a repeating 6-column pattern:

$$\text{Even rows } (r \bmod 2 = 0): [R, G, B, B, G, R] \text{ (period 6)}$$
$$\text{Odd rows } (r \bmod 2 = 1): [B, G, R, R, G, B] \text{ (period 6)}$$

**Theorem 1 (Hexagonal Balance).** Any 6 triangles surrounding a common vertex form a regular hexagon containing exactly 2R + 2G + 2B.

*Proof.* The six triangles surrounding vertex $(vr, vc)$ are at grid positions $(vr-1, vc-1)$, $(vr-1, vc+1)$, $(vr, vc-2)$, $(vr, vc+2)$, $(vr+1, vc-1)$, $(vr+1, vc+1)$. Applying the channel assignment formula yields exactly two of each channel. $\square$

**Theorem 2 (Neighbor Diversity).** For any interior triangle, its three edge-sharing neighbors possess three distinct color channels $\{R, G, B\}$.

This property ensures that the neighbor-borrowing reconstruction (Section 3.1) produces a full RGB value at every triangle position.

### 2.3 Sierpinski Self-Similarity

The equilateral triangle grid admits a natural recursive subdivision: each triangle of side $S$ decomposes into 4 sub-triangles of side $S/2$ (3 of the same orientation at the corners, 1 of opposite orientation at the center). Crucially, the channel assignment pattern is preserved across all scales (verified computationally up to level 5 at 100% accuracy, 11,346 hexagonal groups)—applying the same global formula to fine-grid coordinates $(2r + dr, 2c + dc)$ yields a consistent channel assignment. This fractal property enables multi-scale processing without Gaussian kernel convolution (Section 4).

## 3. Image Reconstruction Pipeline

### 3.1 RAW and Neighbor Borrowing

Given a scene image, each triangle $(r,c)$ captures a single-channel measurement $m(r,c) = I(c^*_{rc}, p_{rc})$, where $c^*_{rc}$ is the assigned channel and $p_{rc}$ is the triangle center.

**Layer 1 (RAW).** The single-channel measurements are directly output as a triangular mosaic. When viewed at sufficient distance, the hexagonal 2R+2G+2B grouping produces a perceptually full-color image without any computation—enabling zero-latency preview at one-third the bandwidth of RGB.

**Layer 2 (Borrow).** Each triangle borrows its two missing channels from its three neighbors. Since the three neighbors possess $\{R, G, B\}$ (Theorem 2), each missing channel has exactly one direct source neighbor:

$$\hat{I}(ch, p_T) = m_{N_{ch}} \quad \text{for } ch \neq c^*_T$$

where $N_{ch}$ is the neighbor whose assigned channel is $ch$. This produces a full RGB estimate at every triangle, but introduces spatial offset artifacts at edges (analogous to Bayer zipper artifacts).

### 3.2 Triangular ISP Correction

We introduce a triangular-native demosaicing algorithm that operates in color-difference space without borrowing concepts from Bayer ISP (which relies on 4-directional gradient selection unavailable on the 3-neighbor triangular graph).

**Algorithm 1: Triangular ISP Correction**

1. Compute initial borrowed RGB for all triangles
2. For $iter = 1$ to $K$:
   a. Compute color differences $\Delta_{RG}, \Delta_{RB}, \Delta_{GB}$ for each triangle
   b. Apply 3-neighbor median filter to each difference map (removes zipper artifacts)
   c. For each triangle, estimate missing channel via edge-weighted average of neighbor color differences:
   
   $$\Delta_{ch}(p_T) = \frac{\sum_{N \in \mathcal{N}(T)} w(T,N) \cdot \Delta_{ch}(p_N)}{\sum_{N} w(T,N)}$$
   
   where $w(T,N) = \exp(-|I_T - I_N|^2 / 2\sigma^2)$ is the edge-avoidance weight
   d. Reconstruct: $I(ch, p_T) = I(c^*_T, p_T) \pm \Delta_{ch}(p_T)$
   e. Known channel remains fixed (ground truth at its own position)

The key difference from Bayer demosaicing: there is no "direction selection" because each missing channel has exactly one source neighbor. Instead, we detect whether the source direction crosses an edge and blend with color-difference estimates from non-edge directions.

### 3.3 AI End-to-End Demosaicing

We train a 3-layer Graph Convolutional Network to directly predict RGB from RAW in a single forward pass, replacing both the borrow and ISP steps:

$$\mathbf{h}^{(l+1)} = \text{ReLU}\left(\frac{\mathbf{X}^{(l)} + \text{mean}_{j\in\mathcal{N}(i)}\mathbf{X}^{(l)}_j}{2} \mathbf{W}^{(l)} + \mathbf{b}^{(l)}\right)$$

The GCN operates on the triangular mesh's natural adjacency (each node has exactly 3 neighbors). Input features are $[value/255, \mathbb{1}_R, \mathbb{1}_G, \mathbb{1}_B]$; output is $[R, G, B] \in [0,1]$. Training uses sensor simulator-generated (noisy RAW, clean RGB) pairs with MSE loss and Adam optimizer.

## 4. Multi-Scale Processing

The Sierpinski subdivision enables a natural multi-scale pyramid without Gaussian blurring:

$$\text{Level } \ell: \text{ side } = S \cdot 2^\ell, \quad \text{grid } = (R/2^\ell, C/2^\ell)$$

**Super-resolution** uses bilinear upsampling of the coarse triangle colors to a finer grid, followed by edge-aware sharpening that preserves color discontinuities at parent-triangle boundaries.

**Multi-scale edge detection** detects edges at each pyramid level, then propagates coarse-scale edges to fine scales: a fine-scale edge is retained only if it has coarse-scale support in its neighborhood. This suppresses texture edges while preserving structural boundaries.

## 5. Triangular Computer Vision Primitives

### 5.1 Edge Detection

The three edge directions of each triangle define natural gradient operators. The gradient magnitude at triangle $T$ is:

$$G(T) = \max_{N \in \mathcal{N}(T)} |I_T - I_N|$$

We apply non-maximum suppression along the gradient direction (comparing $T$ with its neighbor in that direction), followed by double thresholding with connectivity tracing—a triangular analogue of the Canny detector.

### 5.2 Harris Corner Detection

The 2D structure tensor is constructed from the three directional gradients projected to $(g_x, g_y)$:

$$g_x = \frac{\sqrt{3}}{2}(g_2 - g_1), \quad g_y = \pm\left(\frac{1}{2}g_1 + \frac{1}{2}g_2 + g_3\right)$$

The Harris response $R = \det(M) - k \cdot \text{tr}(M)^2$ detects corners on the triangular mesh. Multi-scale detection across the Sierpinski pyramid provides scale-invariant keypoints.

### 5.3 16D Rotation-Invariant Descriptor

A fixed 1-ring template (center triangle + 3 neighbors = 4 triangles, each contributing normalized luminance + 3 edge gradients) yields a 16-dimensional descriptor. Rotation invariance is achieved by cyclically permuting edge indices to align with the dominant gradient direction, exploiting the triangular grid's 3-fold rotational symmetry.

## 6. 3D Unification

Each triangle face maps directly to a 3D mesh face by assigning a depth value to each vertex. The shared-vertex structure of the triangular lattice means adjacent 2D triangles share 3D edges, producing a watertight mesh without additional computation. The mesh can be exported to standard OBJ format, rendered from arbitrary viewpoints, and used as input for 3D computer vision tasks.

Depth-aware edge detection modifies the gradient computation to respect depth discontinuities: edges are only detected where both color and depth gradients agree, suppressing false edges at shadow boundaries.

## 7. Experiments

### 7.1 Experimental Setup

Tests were conducted on a synthetic test suite of 8 image types at 400×400 pixels: directional edges (0°, 45°, 90°, 135°), color boundaries (red-blue), a Siemens star resolution target, a grayscale ramp, a textured pattern, and one real photograph. The triangular sensor was simulated with optical blur ($\sigma=0.5$ px), photon shot noise (ISO 100), and read noise (3 e⁻). Bayer comparison used bilinear demosaicing on the same pixel grid.

All experiments ran on an AMD 7840 CPU with integrated graphics. The pipeline uses numba JIT compilation for 54× acceleration of core loops.

### 7.2 Sensor Efficiency

**Table 1: Triangle vs Bayer sensor comparison (S=12, 2% data)**

| Test Image | TRI PSNR | TRI SSIM | Bayer PSNR |
|-----------|----------|----------|------------|
| Edge 0° | 26.3 dB | 0.983 | 40.3 dB |
| Edge 45° | 27.4 dB | 0.986 | 40.3 dB |
| Edge 90° | 27.9 dB | 0.988 | 40.4 dB |
| Edge 135° | 25.8 dB | 0.981 | 40.3 dB |
| Color R→B | 27.3 dB | 0.988 | 39.3 dB |
| Real Photo | 22.3 dB | 0.905 | 35.4 dB |
| Gray Ramp | 36.4 dB | 0.999 | 42.0 dB |
| Siemens Star | 14.6 dB | 0.477 | 22.2 dB |

**Key observations:**
- Directional anisotropy is only **2.1 dB** (max − min PSNR across edge orientations)
- The 90° edge (aligned with the triangular grid's horizontal axis) achieves the best PSNR
- On smooth gradients, the triangle sensor is nearly lossless (36.4 dB, SSIM 0.999)
- Real photograph achieves SSIM 0.905 with only 2% of Bayer's pixel count
- Siemens star is challenging for both sensors due to high-frequency content

### 7.3 AI vs Hand-Crafted ISP

**Table 2: GCN end-to-end demosaicing vs traditional ISP (S=16)**

| Image | GCN PSNR | ISP PSNR | Δ | GCN Training |
|-------|----------|----------|---|-------------|
| Edge | 24.7 dB | 25.1 dB | −0.4 dB | 0.6s |
| Real Photo | 20.0 dB | 21.4 dB | −1.4 dB | 7.4s |

The 3-layer GCN (~1,300 parameters) approaches hand-crafted ISP quality with a single forward pass. On the edge image, the gap is 0.4 dB. The GCN requires per-image training (self-supervised, 0.6-7.4s). It is a proof-of-concept for learned triangular demosaicing, not yet a general-purpose solution.

### 7.4 Processing Speed

**Table 3: Pipeline throughput (numba JIT, AMD 7840 CPU)**

| Triangle Side | Triangles | Time | FPS |
|--------------|-----------|------|-----|
| S=16 | 1,219 | 17 ms | 59 |
| S=20 | 817 | 12 ms | 85 |
| S=24 | 576 | 8 ms | 121 |
| S=32 | 336 | 5 ms | 203 |

The full pipeline (sample + borrow + ISP correction ×3 iterations + render) achieves real-time performance on consumer hardware, enabled by numba JIT compilation providing 54× speedup over pure Python.

## 8. Discussion

### 8.1 Comparison with Bayer

At equivalent data rates (both RAW formats use 8 bits/pixel), the triangular sensor achieves 70% of Bayer PSNR using only 2% of the spatial samples. The efficiency gap widens as triangle size decreases: at S=4 (15% of Bayer pixel count), the PSNR gap narrows to approximately 7 dB. The triangular sensor's advantage lies not in raw PSNR at equal resolution, but in data efficiency—achieving comparable perceptual quality with dramatically fewer measurements.

### 8.2 Limitations

- **High-frequency loss**: The Siemens star benchmark reveals significant detail loss at high spatial frequencies, inherent to the coarser sampling grid
- **Comparison fairness:** Our efficiency metric compares a triangular sensor at lower spatial sampling density against a full-resolution Bayer sensor with bilinear demosaicing---a simple baseline. A rigorous efficiency proof requires same-photodiode-count comparison against state-of-the-art demosaicing methods (e.g., Hirakawa-Parks, deep joint demosaicking-denoising), deferred to future work on physical sensor prototypes.
- **High-frequency loss:** The Siemens star benchmark reveals predictable detail loss at high spatial frequencies. Frequency sweep analysis (S=12, Nyquist ≈0.083 cy/px) shows PSNR declining from 31.6 dB at 1 cycle to 10.0 dB at 16 cycles, consistent with sparse sampling theory.
- **AI training requirement**: The GCN requires per-image training; a generalizable model would need diverse training data
- **Rendering overhead**: PIL-based triangle rendering is the pipeline bottleneck; GPU rasterization would enable higher resolutions
- **Physical validation**: All results are simulation-based; a physical triangular CMOS sensor is needed for real-world validation

### 8.3 Future Work

The triangular representation's self-similarity and 3D affinity open several research directions:
- **Physical sensor fabrication**: The simulation results motivate CMOS design with triangular photodiode arrangement
- **Video pipeline**: Temporal consistency across triangular frames, exploiting the fixed topology for motion estimation
- **Learned multi-scale super-resolution**: Training larger GCNs on the Sierpinski pyramid for detail synthesis at finer scales
- **Triangular SLAM**: Direct 3D reconstruction from triangular keypoints without 2D-to-3D conversion

## 9. Conclusion

We have presented Triangle Pixel, a complete vision system that replaces rectangular pixel grids with equilateral triangle meshes. The system spans the full pipeline from sensor simulation through AI-native processing, demonstrating that triangular representations can match or exceed rectangular ones in data efficiency, geometric isotropy, and 3D integration, while enabling new capabilities such as zero-computation color reconstruction and Sierpinski-based multi-scale processing. The experimental results validate the approach: 70% of Bayer PSNR at 2% data, 2.1 dB directional anisotropy, and real-time performance at 59 FPS.

---

## References

[1] B. E. Bayer, "Color imaging array," U.S. Patent 3,971,065, 1976.

[2] J. F. Hamilton and J. E. Adams, "Adaptive color plan interpolation in single sensor color electronic camera," U.S. Patent 5,629,734, 1997.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," CVPR 2016.

[4] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," ICLR 2017.

[5] D. G. Lowe, "Distinctive image features from scale-invariant keypoints," IJCV 2004.

[6] C. Harris and M. Stephens, "A combined corner and edge detector," Alvey Vision Conference 1988.

[7] J. Canny, "A computational approach to edge detection," IEEE TPAMI 1986.
