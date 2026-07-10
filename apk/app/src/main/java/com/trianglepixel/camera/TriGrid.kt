package com.trianglepixel.camera

import kotlin.math.sqrt

/**
 * Triangular grid geometry - ported from triangle_engine.py
 *
 * Grid: equilateral triangles of side S, height h = S * sqrt(3) / 2.
 * Tri (r,c) center: x = c * S/2, y = r * h + (upward? 2h/3 : h/3)
 * Channel: 6-column period, 2R+2G+2B per hexagon.
 */
object TriGrid {

    /** Triangle side length in pixels */
    var S: Float = 16f
    /** Triangle height */
    val h: Float get() = (S * sqrt(3.0) / 2.0).toFloat()

    /** Even-row channel pattern [R,G,B,B,G,R] -> 0,1,2,2,1,0 */
    private val evenRow = intArrayOf(0, 1, 2, 2, 1, 0)
    /** Odd-row channel pattern [B,G,R,R,G,B] -> 2,1,0,0,1,2 */
    private val oddRow  = intArrayOf(2, 1, 0, 0, 1, 2)

    /** Which color channel (0=R, 1=G, 2=B) does triangle (r,c) sense? */
    fun assignedChannel(r: Int, c: Int): Int {
        return if (r % 2 == 0) evenRow[c % 6] else oddRow[c % 6]
    }

    /** Is triangle (r,c) pointing upward? (apex at top) */
    fun isUpward(r: Int, c: Int): Boolean = (r + c) % 2 == 0

    /**
     * Center (x, y) of triangle (r, c) in sensor pixel coordinates.
     * Upward triangle: center is at 2h/3 from top.
     * Downward triangle: center is at h/3 from top.
     */
    fun triCenter(r: Int, c: Int): Pair<Float, Float> {
        val x = c * S / 2f
        val y = r * h + if (isUpward(r, c)) 2f * h / 3f else h / 3f
        return x to y
    }

    /**
     * Three edge-sharing neighbors of triangle (r,c).
     *
     * Upward (apex up): neighbors are left, right, bottom.
     * Downward (apex down): neighbors are left, right, top.
     *
     * Returns list of (r, c) pairs. Out-of-bounds => (-1, -1).
     */
    fun neighborsOf(r: Int, c: Int, nRows: Int, nCols: Int): List<Pair<Int, Int>> {
        val result = mutableListOf<Pair<Int, Int>>()

        // Horizontal neighbors (same row)
        result.add(r to (c - 1).coerceIn(0, nCols - 1))
        result.add(r to (c + 1).coerceIn(0, nCols - 1))

        // Vertical neighbor
        val vr = if (isUpward(r, c)) r + 1 else r - 1
        result.add(vr.coerceIn(0, nRows - 1) to c)

        return result
    }

    /**
     * Compute grid dimensions from sensor resolution.
     * nCols: number of triangle columns fitting in width.
     * nRows: number of triangle rows fitting in height.
     */
    fun computeGrid(imageW: Int, imageH: Int): Pair<Int, Int> {
        val nCols = (imageW / (S / 2f)).toInt() + 3
        val nRows = (imageH / h).toInt() + 2
        return nRows to nCols
    }

    /** Build adjacency matrix [N, 3] as flat index array */
    fun buildAdjacency(nRows: Int, nCols: Int): Array<IntArray> {
        val N = nRows * nCols
        val adj = Array(N) { IntArray(3) { -1 } }
        for (r in 0 until nRows) {
            for (c in 0 until nCols) {
                val idx = r * nCols + c
                val nbrs = neighborsOf(r, c, nRows, nCols)
                for (k in nbrs.indices) {
                    val (nr, nc) = nbrs[k]
                    adj[idx][k] = nr * nCols + nc
                }
            }
        }
        return adj
    }
}
