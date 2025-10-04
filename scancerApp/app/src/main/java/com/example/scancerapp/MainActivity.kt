package com.example.scancerapp

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.example.scancerapp.ui.theme.ScancerAppTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

// Data class untuk menampung hasil inferensi
data class InferenceResult(val diagnosis: String, val computationTime: Long)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ScancerAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    DiagnosisScreen()
                }
            }
        }
    }
}

@Composable
fun DiagnosisScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var imageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var diagnosisResult by remember { mutableStateOf<InferenceResult?>(null) } // Perbaikan di sini

    val model: Module? by remember { mutableStateOf(loadModel(context, "model_mobile_lite.ptl")) }
    val labels: List<String>? by remember { mutableStateOf(loadLabels(context, "labels.txt")) }

    var tempUri: Uri? by remember { mutableStateOf(null) }

    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture(),
        onResult = { success ->
            val currentTempUri = tempUri
            if (success && currentTempUri != null) {
                try {
                    context.contentResolver.openInputStream(currentTempUri)?.use { inputStream ->
                        imageBitmap = BitmapFactory.decodeStream(inputStream)
                        diagnosisResult = null
                    }
                } catch (e: Exception) {
                    Log.e("TakePicture", "Error processing image from camera", e)
                    Toast.makeText(context, "Gagal memproses gambar", Toast.LENGTH_SHORT).show()
                }
            } else if (success && currentTempUri == null) {
                Log.e("TakePicture", "Picture taken successfully but tempUri is null")
                Toast.makeText(context, "Gagal mendapatkan URI gambar", Toast.LENGTH_SHORT).show()
            }
        }
    )

    val pickMediaLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri ->
            if (uri != null) {
                try {
                    context.contentResolver.openInputStream(uri)?.use { inputStream ->
                        imageBitmap = BitmapFactory.decodeStream(inputStream)
                        diagnosisResult = null
                    }
                } catch (e: Exception) {
                    Log.e("PickMedia", "Error processing image from gallery", e)
                    Toast.makeText(context, "Gagal memproses gambar dari galeri", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(context, "Tidak ada media dipilih", Toast.LENGTH_SHORT).show()
            }
        }
    )

    val requestCameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            val newUri = createImageFileUri(context)
            if (newUri != null) {
                tempUri = newUri
                takePictureLauncher.launch(newUri)
            } else {
                Toast.makeText(context, "Gagal membuat file gambar", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(context, "Izin kamera ditolak", Toast.LENGTH_SHORT).show()
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        imageBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Gambar Terpilih",
                modifier = Modifier
                    .size(250.dp)
                    .padding(bottom = 16.dp)
            )
        } ?: Box(
            modifier = Modifier
                .size(250.dp)
                .padding(bottom = 16.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("Pilih atau ambil gambar")
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = {
                requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }) {
                Text("Ambil Gambar")
            }
            Button(onClick = {
                pickMediaLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }) {
                Text("Unggah Gambar")
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        if (imageBitmap != null && model != null) {
            Button(onClick = {
                val currentImageBitmap = imageBitmap!!
                val currentModel = model!!

                CoroutineScope(Dispatchers.IO).launch {
                    val result = runInference(currentModel, currentImageBitmap, labels)
                    withContext(Dispatchers.Main) {
                        diagnosisResult = result
                    }
                }
            }) {
                Text("Diagnosis")
            }
        } else if (imageBitmap != null && model == null) {
            Text("Model tidak berhasil dimuat. Periksa Logcat untuk error.", color = MaterialTheme.colorScheme.error)
        }

        Spacer(modifier = Modifier.height(20.dp))

        diagnosisResult?.let {
            Text(
                text = "Diagnosis: ${it.diagnosis}", // Menampilkan diagnosis
                style = MaterialTheme.typography.headlineSmall
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Waktu Komputasi: ${it.computationTime} ms", // Menampilkan waktu komputasi
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}

fun createImageFileUri(context: Context): Uri? {
    val timeStamp = System.currentTimeMillis()
    val imageFileName = "JPEG_${timeStamp}_"
    val storageDir: File? = context.cacheDir

    if (storageDir == null) {
        Log.e("FileCreation", "Cache directory is null.")
        return null
    }

    return try {
        val imageFile = File.createTempFile(
            imageFileName,
            ".jpg",
            storageDir
        )
        FileProvider.getUriForFile(
            context,
            "${context.packageName}.provider",
            imageFile
        )
    } catch (ex: IOException) {
        Log.e("FileCreation", "Error creating image file", ex)
        null
    }
}

fun loadModel(context: Context, modelName: String): Module? {
    return try {
        val modelPath = assetFilePath(context, modelName)
        Log.i("PytorchLoad", "Attempting to load model from: $modelPath")
        val module = LiteModuleLoader.load(modelPath)
        Log.i("PytorchLoad", "Model loaded successfully: $modelName")
        module
    } catch (e: Exception) {
        Log.e("PytorchLoad", "Error loading model '$modelName': ${e.message}", e)
        Toast.makeText(context, "Gagal memuat model: ${e.message}", Toast.LENGTH_LONG).show()
        null
    }
}

@Throws(IOException::class)
fun assetFilePath(context: Context, assetName: String): String {
    val file = File(context.filesDir, assetName)
    if (!file.exists() || file.length() == 0L) {
        Log.i("AssetUtil", "File $assetName not found or empty in app's internal storage. Copying from assets.")
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        Log.i("AssetUtil", "File $assetName copied to ${file.absolutePath}")
    } else {
        Log.i("AssetUtil", "File $assetName already exists in app's internal storage: ${file.absolutePath}")
    }
    return file.absolutePath
}

fun loadLabels(context: Context, fileName: String): List<String>? {
    return try {
        Log.i("PytorchLoad", "Attempting to load labels from: $fileName")
        val labels = context.assets.open(fileName).bufferedReader().useLines { lines ->
            lines.toList()
        }
        Log.i("PytorchLoad", "Labels loaded successfully: ${labels.size} labels found.")
        labels
    } catch (e: IOException) {
        Log.e("PytorchLoad", "Error loading labels '$fileName': ${e.message}", e)
        Toast.makeText(context, "Gagal memuat label: ${e.message}", Toast.LENGTH_LONG).show()
        null
    }
}

fun preprocessImage(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): org.pytorch.Tensor {
    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    val MEAN_RGB = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD_RGB = floatArrayOf(0.229f, 0.224f, 0.225f)
    return TensorImageUtils.bitmapToFloat32Tensor(
        resizedBitmap, MEAN_RGB, STD_RGB
    )
}

fun runInference(model: Module, bitmap: Bitmap, labels: List<String>?): InferenceResult {
    return try {
        val startTime = System.currentTimeMillis() // Catat waktu mulai

        val inputWidth = 224
        val inputHeight = 224
        val inputTensor = preprocessImage(bitmap, inputWidth, inputHeight)

        Log.i("PytorchRun", "Running inference...")
        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()

        val computationTime = System.currentTimeMillis() - startTime // Hitung selisih waktu

        val scores = outputTensor.dataAsFloatArray
        Log.i("PytorchRun", "Inference completed in $computationTime ms. Scores: ${scores.joinToString()}")

        if (scores.isEmpty()) {
            return InferenceResult("Error: Model output is empty.", computationTime)
        }

        var maxScoreIdx = -1
        var maxScore = -Float.MAX_VALUE
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxScoreIdx = i
            }
        }

        if (maxScoreIdx == -1) {
            return InferenceResult("Error: Could not determine class from scores.", computationTime)
        }

        val resultText = labels?.getOrNull(maxScoreIdx) ?: "Kelas $maxScoreIdx"
        val formattedScore = String.format("%.2f", maxScore)
        val diagnosis = "$resultText (Skor: $formattedScore)"

        InferenceResult(diagnosis, computationTime)

    } catch (e: Exception) {
        Log.e("PytorchRun", "Error during inference: ${e.message}", e)
        InferenceResult("Error saat inferensi: ${e.message}", -1)
    }
}

@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    ScancerAppTheme {
        DiagnosisScreen()
    }
}
