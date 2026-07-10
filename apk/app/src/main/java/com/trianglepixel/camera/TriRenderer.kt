package com.trianglepixel.camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path

/**
 * Render triangular grid RGB values to a display Bitmap.
 *
 * Each triangle is drawn as a filled polygon with its reconstructed RGB color.
 * This produces the final viewable image from triangular data.
 */
object TriRenderer {

    /**
     * Render triangles to Bitmap at display resolution.
     *
     * @param rgb      FloatArray [nRows * nCols * 3] of RGB values (0-255 or 0-1023)
     * @param nRows    Triangle grid rows
     * @param nCols    Triangle grid columns
     * @param outW     Output bitmap width
     * @param outH     Output bitmap height
     * @param maxVal   Maximum RAW value (255 for YUV, 1023 for 10-bit)
     */
    fun render(
        rgb: FloatArray,
        nRows: Int, nCols: Int,
        outW: Int, outH: Int,
        maxVal: Float = 255f
    ): Bitmap {
        val bitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        val path = Path()

        val S = TriGrid.S
        val h = TriGrid.h

        for (r in 0 until nRows) {
            for (c in 0 until nCols) {
                val idx = r * nCols + c
                val out = idx * 3

                // Clamp RGB to [0, 255]
                val red   = ((rgb[out]     / maxVal).coerceIn(0f, 1f) * 255).toInt()
                val green = ((rgb[out + 1] / maxVal).coerceIn(0f, 1f) * 255).toInt()
                val blue  = ((rgb[out + 2] / maxVal).coerceIn(0f, 1f) * 255).toInt()

                paint.color = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue

                // Triangle vertices in sensor coordinates
                val cx = c * S / 2f
                val cy = r * h + if (TriGrid.isUpward(r, c)) h / 3f else 2f * h / 3f

                path.reset()
                if (TriGrid.isUpward(r, c)) {
                    // Apex up: left bottom, right bottom, top center
                    path.moveTo(cx - S / 2f, cy + 2f * h / 3f)
                    path.lineTo(cx + S / 2f, cy + 2f * h / 3f)
                    path.lineTo(cx, cy - h / 3f)
                } else {
                    // Apex down: left top, right top, bottom center
                    path.moveTo(cx - S / 2f, cy - h / 3f)
                    path.lineTo(cx + S / 2f, cy - h / 3f)
                    path.lineTo(cx, cy + 2f * h / 3f)
                }
                path.close()
                canvas.drawPath(path, paint)
            }
        }
        return bitmap
    }
}
