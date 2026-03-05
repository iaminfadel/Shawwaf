# MiniCV — Math & Algorithm Notes

Mathematical foundations and algorithm descriptions for each major technique implemented in the MiniCV library.

---

## 1. Color Conversion (Grayscale)

RGB to grayscale using the **Rec. 601 luminance** formula:

$$Y = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$$

These weights reflect the human eye's sensitivity: green contributes most to perceived brightness, followed by red, then blue.

---

## 2. 2D Convolution

Discrete 2D convolution of an image $f$ with kernel $h$:

$$(f * h)[m, n] = \sum_{i} \sum_{j} f[m - i, n - j] \cdot h[i, j]$$

**Key properties:**
- The kernel is **flipped** (rotated 180°) before sliding (true convolution, not correlation).
- Boundary handling uses the `pad()` function (zero, reflect, or replicate).
- Output has the same spatial size as the input.

Our implementation uses vectorized shifted-view accumulation rather than per-pixel looping.

---

## 3. Gaussian Kernel

The 2D Gaussian kernel centered at the origin:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

The discrete kernel is sampled on a grid $[-k, k] \times [-k, k]$ where $k = \lfloor\text{size}/2\rfloor$, then normalized so all entries sum to 1.

---

## 4. Median Filter

The median filter replaces each pixel with the **median** of its local neighborhood. It is a non-linear filter and thus cannot be implemented via convolution.

$$\text{output}[m, n] = \text{median}\{f[m+i, n+j] \mid (i,j) \in \mathcal{N}\}$$

where $\mathcal{N}$ is the neighborhood footprint. This filter excels at removing **salt-and-pepper noise** without blurring edges.

---

## 5. Thresholding

### 5.1 Global Threshold

Simple binarization with a user-supplied threshold $T$:

$$B[m,n] = \begin{cases} 255 & \text{if } f[m,n] \geq T \\\\ 0 & \text{otherwise} \end{cases}$$

### 5.2 Otsu's Method

Finds the threshold $T^*$ that **maximizes inter-class variance**:

$$T^* = \arg\max_T \; \omega_0(T) \cdot \omega_1(T) \cdot [\mu_0(T) - \mu_1(T)]^2$$

where $\omega_0, \omega_1$ are the class weights (proportions of pixels below/above $T$) and $\mu_0, \mu_1$ are the class means. Evaluated exhaustively over all 256 possible thresholds.

### 5.3 Adaptive Threshold

Local thresholding computes a per-pixel threshold from a neighborhood average:

$$B[m,n] = \begin{cases} 255 & \text{if } f[m,n] > \bar{f}_{\mathcal{N}}[m,n] - C \\\\ 0 & \text{otherwise} \end{cases}$$

where $\bar{f}_{\mathcal{N}}$ is the local mean (or Gaussian-weighted mean) in a window of size `block_size`, and $C$ is a constant offset.

---

## 6. Sobel Edge Detection

The 3×3 Sobel kernels compute approximate image gradients:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \end{bmatrix} * f, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \end{bmatrix} * f$$

Derived quantities:
- **Magnitude:** $M = \sqrt{G_x^2 + G_y^2}$
- **Direction:** $\theta = \arctan_2(G_y, G_x)$

---

## 7. Histogram Equalization

CDF-based contrast enhancement for a grayscale uint8 image:

1. Compute the histogram $h[k]$ for $k = 0, \ldots, 255$.
2. Compute the cumulative distribution: $\text{CDF}[k] = \sum_{i=0}^{k} h[i]$.
3. Build the look-up table:

$$\text{LUT}[k] = \text{round}\!\left(\frac{\text{CDF}[k] - \text{CDF}_{\min}}{N - \text{CDF}_{\min}} \cdot 255\right)$$

where $N$ is the total pixel count and $\text{CDF}_{\min}$ is the minimum non-zero CDF value.

---

## 8. Bit-Plane Slicing

For a uint8 image, bit plane $p \in [0, 7]$ is extracted as:

$$B_p[m,n] = \left\lfloor \frac{f[m,n]}{2^p} \right\rfloor \mod 2$$

Plane 0 is the LSB (most noise-like), plane 7 is the MSB (most structure).

---

## 9. Laplacian Sharpening

The Laplacian is a second-derivative operator that highlights intensity discontinuities:

$$\nabla^2 f = \begin{bmatrix} 0 & -1 & 0 \\\\ -1 & 4 & -1 \\\\ 0 & -1 & 0 \end{bmatrix} * f$$

Sharpening adds the Laplacian back to the original image:

$$f_{\text{sharp}} = f + \alpha \cdot \nabla^2 f$$

where $\alpha$ is the strength parameter.

---

## 10. Gamma Correction

Power-law (gamma) transform adjusts image brightness:

$$g[m,n] = 255 \cdot \left(\frac{f[m,n]}{255}\right)^\gamma$$

- $\gamma < 1$: brightens dark regions (expands low intensities).
- $\gamma > 1$: darkens bright regions (compresses high intensities).
- $\gamma = 1$: identity transform.

---

## 11. Geometric Transforms

### 11.1 Resize (Interpolation)

**Nearest-neighbor:** Map each target pixel to the closest source pixel.

**Bilinear:** Compute a weighted average of the four nearest source pixels:

$$f(x, y) = (1-a)(1-b)\,f_{00} + a(1-b)\,f_{10} + (1-a)b\,f_{01} + ab\,f_{11}$$

where $a, b$ are the fractional distances from the top-left neighbor.

### 11.2 Rotation

Inverse mapping: for each output pixel $(u, v)$, compute the source coordinates:

$$\begin{bmatrix} x \\\\ y \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} u - c_x \\\\ v - c_y \end{bmatrix} + \begin{bmatrix} c_x \\\\ c_y \end{bmatrix}$$

where $(c_x, c_y)$ is the center of rotation.

---

## 12. HOG (Histogram of Oriented Gradients)

1. Compute gradient magnitude and direction via Sobel.
2. Divide image into non-overlapping cells of `cell_size × cell_size`.
3. In each cell, accumulate a magnitude-weighted histogram of gradient orientations (bins over $[0°, 180°)$).
4. Concatenate all cell histograms into a single feature vector.
5. L2-normalize the vector.

---

## 13. Bresenham's Line Algorithm

Integer-only line rasterization. Tracks an error variable to decide when to step in the secondary axis:

```
err = dx - dy
while not at endpoint:
    plot(x0, y0)
    e2 = 2 * err
    if e2 > -dy: err -= dy; x0 += sx
    if e2 <  dx: err += dx; y0 += sy
```

Produces the optimal integer approximation of a line segment.
