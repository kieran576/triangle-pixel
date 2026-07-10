package com.trianglepixel.camera

/**
 * Bayer RAW → triangular grid sampler.
 *
 * Reads Bayer RAW pixel data and samples each triangle's
 * assigned color channel from the nearest Bayer pixels.
 */
object TriSampler {

    /**
     * Sample triangular grid from Bayer RAW byte array.
     *
     * @param bayerData  RAW byte array (Image.Plane buffer)
     * @param bayerW     Bayer image width
     * @param bayerH     Bayer image height
     * @param rowStride  Row stride in bytes
     * @param pixelStride Pixel stride in bytes
     * @param nRows      Triangle grid rows
     * @param nCols      Triangle grid columns
     * @return FloatArray [nRows * nCols] of RAW values [0-1023] (10-bit) or [0-4095] (12-bit)
     */
    fun sampleBayer(
        bayerData: ByteArray,
        bayerW: Int, bayerH: Int,
        rowStride: Int, pixelStride: Int,
        nRows: Int, nCols: Int
    ): FloatArray {
        val raw = FloatArray(nRows * nCols)

        for (r in 0 until nRows) {
            for (c in 0 until nCols) {
                val (cx, cy) = TriGrid.triCenter(r, c)
                val targetCh = TriGrid.assignedChannel(r, c)

                // Sample target channel from nearest Bayer pixel
                val val10bit = sampleChannelAt(
                    bayerData, bayerW, bayerH,
                    rowStride, pixelStride,
                    cx.toInt(), cy.toInt(), targetCh
                )
                raw[r * nCols + c] = val10bit.toFloat()
            }
        }
        return raw
    }

    /**
     * Sample a specific color channel at pixel position (px, py)
     * from a Bayer RGGB sensor.
     *
     * Bayer pattern (top-left = R):
     *   Even row: R G R G ...
     *   Odd row:  G B G B ...
     *
     * Returns the RAW value (10 or 12 bit) from the nearest
     * Bayer pixel of the target channel.
     */
    private fun sampleChannelAt(
        data: ByteArray,
        w: Int, h: Int,
        rowStride: Int, pixelStride: Int,
        px: Int, py: Int,
        targetCh: Int  // 0=R, 1=G, 2=B
    ): Int {
        // Search in a small neighborhood for the target channel
        val radius = 2
        var sum = 0
        var count = 0

        for (dy in -radius..radius) {
            val sy = (py + dy).coerceIn(0, h - 1)
            for (dx in -radius..radius) {
                val sx = (px + dx).coerceIn(0, w - 1)
                // Determine Bayer channel at (sx, sy)
                val bayCh = if (sy % 2 == 0) {
                    if (sx % 2 == 0) 0 else 1  // even row: R, G
                } else {
                    if (sx % 2 == 0) 1 else 2  // odd row: G, B
                }
                if (bayCh == targetCh) {
                    val offset = sy * rowStride + sx * pixelStride
                    if (offset + 1 < data.size) {
                        // 10-bit: read 2 bytes, unpack
                        val lo = data[offset].toInt() and 0xFF
                        val hi = data[offset + 1].toInt() and 0xFF
                        sum += (hi shl 8) or lo
                        count++
                    }
                }
            }
        }
        return if (count > 0) sum / count else 0
    }

    /**
     * Simplified: sample from YUV_420_888 (fallback when RAW unavailable).
     * Uses Y (luminance) plane as approximate single-channel value.
     */
    fun sampleYuv(
        yPlane: ByteArray,
        yRowStride: Int,
        yW: Int, yH: Int,
        nRows: Int, nCols: Int
    ): FloatArray {
        val raw = FloatArray(nRows * nCols)

        for (r in 0 until nRows) {
            for (c in 0 until nCols) {
                val (cx, cy) = TriGrid.triCenter(r, c)
                val px = cx.toInt().coerceIn(0, yW - 1)
                val py = cy.toInt().coerceIn(0, yH - 1)
                val offset = py * yRowStride + px
                raw[r * nCols + c] = if (offset < yPlane.size) {
                    (yPlane[offset].toInt() and 0xFF).toFloat()
                } else 0f
            }
        }
        return raw
    }
}
