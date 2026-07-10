package com.trianglepixel.camera

/**
 * Neighbor-borrowing RGB reconstruction.
 *
 * Each triangle's RGB = [leftNeighbor, rightNeighbor, verticalNeighbor],
 * own value not used for self, only contributed to neighbors.
 *
 * This is the zero-computation color reconstruction.
 */
object TriBorrow {

    /**
     * Reconstruct RGB from single-channel RAW.
     *
     * @param raw      FloatArray [nRows * nCols] of single-channel values
     * @param nRows    Triangle grid rows
     * @param nCols    Triangle grid columns
     * @return FloatArray [nRows * nCols * 3] of RGB values (R, G, B interleaved)
     */
    fun reconstruct(raw: FloatArray, nRows: Int, nCols: Int): FloatArray {
        val rgb = FloatArray(nRows * nCols * 3)
        val adj = TriGrid.buildAdjacency(nRows, nCols)

        for (r in 0 until nRows) {
            for (c in 0 until nCols) {
                val idx = r * nCols + c
                val ownCh = TriGrid.assignedChannel(r, c)
                val nbrs = adj[idx]

                // left = neighbor 0, right = neighbor 1, vertical = neighbor 2
                val leftVal  = raw[nbrs[0]]
                val rightVal = raw[nbrs[1]]
                val vertVal  = raw[nbrs[2]]

                // Assign to R, G, B based on which neighbor provides which channel
                // left neighbor's channel = assignedChannel(nr0, nc0)
                val nbrRow0 = nbrs[0] / nCols; val nbrCol0 = nbrs[0] % nCols
                val nbrRow1 = nbrs[1] / nCols; val nbrCol1 = nbrs[1] % nCols
                val nbrRow2 = nbrs[2] / nCols; val nbrCol2 = nbrs[2] % nCols

                val ch0 = TriGrid.assignedChannel(nbrRow0, nbrCol0)
                val ch1 = TriGrid.assignedChannel(nbrRow1, nbrCol1)
                val ch2 = TriGrid.assignedChannel(nbrRow2, nbrCol2)

                // Map: neighbor with channel K → RGB[K]
                val out = idx * 3
                rgb[out + ch0] = leftVal
                rgb[out + ch1] = rightVal
                rgb[out + ch2] = vertVal
            }
        }
        return rgb
    }

    /**
     * Fast RAW-direct preview: fill each triangle with its own single-channel value.
     * No borrowing - just show what the sensor sees. This is the zero-latency mode.
     */
    fun rawDirect(raw: FloatArray, nRows: Int, nCols: Int): FloatArray {
        val rgb = FloatArray(nRows * nCols * 3)

        for (r in 0 until nRows) {
            for (c in 0 until nCols) {
                val idx = r * nCols + c
                val ch = TriGrid.assignedChannel(r, c)
                val out = idx * 3
                // Fill only the target channel, leave others at 0
                rgb[out + ch] = raw[idx]
            }
        }
        return rgb
    }
}
