package com.trianglepixel.camera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.Surface
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.atomic.AtomicInteger

/**
 * Main Activity — triangular camera.
 *
 * Uses Camera2 API to capture RAW_SENSOR or YUV frames,
 * converts to triangular grid, and displays in real-time.
 *
 * Modes: RAW direct (zero-latency) / borrow / ISP.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private lateinit var previewView: ImageView
    private lateinit var modeText: TextView
    private lateinit var fpsText: TextView

    // Modes: 0=RAW direct, 1=borrow, 2=ISP corrected
    private val modeIndex = AtomicInteger(1)
    private val modeNames = arrayOf("RAW Direct", "Borrow RGB", "ISP")

    private var frameCount = 0
    private var lastFpsTime = 0L
    private var previewW = 1080
    private var previewH = 1920

    // Triangular grid params
    private var nRows = 0
    private var nCols = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        modeText = findViewById(R.id.modeText)
        fpsText = findViewById(R.id.fpsText)

        findViewById<Button>(R.id.btnMode).setOnClickListener { cycleMode() }
        findViewById<Button>(R.id.btnCapture).setOnClickListener { captureFrame() }

        TriGrid.S = 16f  // default triangle side

        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), 100
            )
        } else {
            startCamera()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    // ============================================================
    //  Camera lifecycle
    // ============================================================

    private fun startCamera() {
        startBackgroundThread()

        try {
            // Pick back camera
            val cameraId = cameraManager.cameraIdList.first { id ->
                cameraManager.getCameraCharacteristics(id)
                    .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            }

            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val streamConfig = characteristics
                .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!

            // Prefer RAW_SENSOR, fallback to YUV
            val rawSizes = streamConfig.getOutputSizes(ImageFormat.RAW_SENSOR)
            val yuvSizes = streamConfig.getOutputSizes(ImageFormat.YUV_420_888)

            val useRaw = rawSizes != null && rawSizes.isNotEmpty()
            val sizes = if (useRaw) rawSizes else yuvSizes!!
            val previewSize = chooseSize(sizes, 1080, 1920)

            previewW = previewSize.width
            previewH = previewSize.height

            // Compute triangular grid
            val (nr, nc) = TriGrid.computeGrid(previewW, previewH)
            nRows = nr; nCols = nc

            val format = if (useRaw) ImageFormat.RAW_SENSOR else ImageFormat.YUV_420_888
            imageReader = ImageReader.newInstance(
                previewW, previewH, format, 2
            )
            imageReader!!.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                processFrame(image, useRaw)
                image.close()
            }, backgroundHandler)

            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) return

            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createCaptureSession()
                }
                override fun onDisconnected(camera: CameraDevice) { camera.close() }
                override fun onError(camera: CameraDevice, error: Int) { camera.close() }
            }, backgroundHandler)

            modeText.text = "Mode: ${modeNames[modeIndex.get()]}"
            Toast.makeText(this, "Grid: ${nRows}x$nCols (${if (useRaw) "RAW" else "YUV"})",
                Toast.LENGTH_SHORT).show()

        } catch (e: Exception) {
            Toast.makeText(this, "Camera error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun createCaptureSession() {
        val device = cameraDevice ?: return
        val surface = imageReader!!.surface

        device.createCaptureSession(listOf(surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    startRepeatingRequest()
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {}
            }, backgroundHandler)
    }

    private fun startRepeatingRequest() {
        val device = cameraDevice ?: return
        val session = captureSession ?: return

        val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        request.addTarget(imageReader!!.surface)
        session.setRepeatingRequest(request.build(), null, backgroundHandler)
    }

    // ============================================================
    //  Frame processing
    // ============================================================

    private fun processFrame(image: android.media.Image, isRaw: Boolean) {
        val t0 = System.currentTimeMillis()

        val raw: FloatArray
        if (isRaw) {
            // RAW_SENSOR: unpack from planes
            val plane = image.planes[0]
            val buffer = plane.buffer
            val data = ByteArray(buffer.remaining())
            buffer.get(data)
            raw = TriSampler.sampleBayer(
                data, image.width, image.height,
                plane.rowStride, plane.pixelStride,
                nRows, nCols
            )
        } else {
            // YUV fallback: use Y plane as luminance
            val yPlane = image.planes[0]
            val buffer = yPlane.buffer
            val data = ByteArray(buffer.remaining())
            buffer.get(data)
            raw = TriSampler.sampleYuv(
                data, yPlane.rowStride,
                image.width, image.height,
                nRows, nCols
            )
        }

        val maxVal = if (isRaw) 1023f else 255f

        // Reconstruct based on mode
        val rgb = when (modeIndex.get()) {
            0 -> TriBorrow.rawDirect(raw, nRows, nCols)  // RAW direct
            1 -> TriBorrow.reconstruct(raw, nRows, nCols)  // borrow
            2 -> TriBorrow.reconstruct(raw, nRows, nCols)  // borrow (ISP simplified)
            else -> TriBorrow.reconstruct(raw, nRows, nCols)
        }

        val bitmap = TriRenderer.render(
            rgb, nRows, nCols, previewW, previewH, maxVal
        )

        val elapsed = System.currentTimeMillis() - t0

        runOnUiThread {
            previewView.setImageBitmap(bitmap)
            updateFps()
        }
    }

    // ============================================================
    //  UI
    // ============================================================

    private fun cycleMode() {
        val next = (modeIndex.get() + 1) % 3
        modeIndex.set(next)
        modeText.text = "Mode: ${modeNames[next]}"
    }

    private fun captureFrame() {
        // Save current preview bitmap to gallery
        val bitmap = (previewView.drawable as? android.graphics.drawable.BitmapDrawable)?.bitmap
        if (bitmap != null) {
            val path = "${getExternalFilesDir(null)}/tri_${System.currentTimeMillis()}.png"
            java.io.FileOutputStream(path).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Toast.makeText(this, "Saved: $path", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateFps() {
        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFpsTime > 1000) {
            val fps = frameCount * 1000f / (now - lastFpsTime)
            fpsText.text = "%.1f FPS".format(fps)
            frameCount = 0
            lastFpsTime = now
        }
    }

    // ============================================================
    //  Helpers
    // ============================================================

    private fun chooseSize(sizes: Array<Size>, targetW: Int, targetH: Int): Size {
        return sizes.minByOrNull { size ->
            kotlin.math.abs(size.width - targetW) + kotlin.math.abs(size.height - targetH)
        } ?: sizes[0]
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBg").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    override fun onDestroy() {
        super.onDestroy()
        captureSession?.close()
        cameraDevice?.close()
        imageReader?.close()
        backgroundThread?.quitSafely()
    }
}
