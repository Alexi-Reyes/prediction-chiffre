const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clearButton');
const predictButton = document.getElementById('predictButton');
const predictionResult = document.getElementById('predictionResult');

const MODEL_PATH = 'image_classifier_model.onnx';
const INPUT_TENSOR_SIZE = 28;
const CANVAS_SIZE = 280;

const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;
const BINARY_THRESHOLD = 128;

let isDrawing = false;
let inferenceSession = null;

ctx.lineWidth = 25;
ctx.lineCap = 'round';
ctx.strokeStyle = '#FFFFFF';

ctx.fillStyle = '#000000';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// charge le model
async function loadModel() {
    try {
        predictionResult.textContent = 'Loading AI model...';
        inferenceSession = await ort.InferenceSession.create(MODEL_PATH);
        console.log("ONNX model loaded successfully.");
        predictionResult.textContent = 'Model Ready. Draw a digit!';
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
        predictionResult.textContent = 'ERROR: Failed to load model. Check console.';
    }
}
loadModel();


// fonction qui transforme le résultat de la canvas pour ressembler le plus possible au model entrainé
function preprocess(canvas) {
    const fullSizeImgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const fullSizePixels = fullSizeImgData.data;

    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = INPUT_TENSOR_SIZE;
    offscreenCanvas.height = INPUT_TENSOR_SIZE;
    const offscreenCtx = offscreenCanvas.getContext('2d');

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = CANVAS_SIZE;
    tempCanvas.height = CANVAS_SIZE;
    const tempCtx = tempCanvas.getContext('2d');

    const tempImgData = tempCtx.createImageData(CANVAS_SIZE, CANVAS_SIZE);
    const tempPixels = tempImgData.data;

    for (let i = 0; i < fullSizePixels.length; i += 4) {
        const rawPixelValue = fullSizePixels[i];

        if (rawPixelValue >= BINARY_THRESHOLD) {
            tempPixels[i] = 255; tempPixels[i + 1] = 255; tempPixels[i + 2] = 255; tempPixels[i + 3] = 255;
        } else {
            tempPixels[i] = 0; tempPixels[i + 1] = 0; tempPixels[i + 2] = 0; tempPixels[i + 3] = 255;
        }
    }
    tempCtx.putImageData(tempImgData, 0, 0);

    console.log("Raw Pixel Check (280x280 image):", tempImgData.data.includes(255));

    offscreenCtx.drawImage(tempCanvas, 0, 0, INPUT_TENSOR_SIZE, INPUT_TENSOR_SIZE);

    const finalImgData = offscreenCtx.getImageData(0, 0, INPUT_TENSOR_SIZE, INPUT_TENSOR_SIZE);
    const finalPixels = finalImgData.data;

    const inputTensorData = new Float32Array(1 * 1 * INPUT_TENSOR_SIZE * INPUT_TENSOR_SIZE);
    let tensorIndex = 0;

    for (let i = 0; i < finalPixels.length; i += 4) {
        const normalized_0_1 = finalPixels[i] / 255.0;

        const finalValue = (normalized_0_1 - MNIST_MEAN) / MNIST_STD;

        inputTensorData[tensorIndex++] = finalValue;
    }


    console.log("Pre-processed tensor start (0=black, 1=white, normalized):", inputTensorData.slice(0, 20));

    const inputTensor = new ort.Tensor('float32', inputTensorData, [1, 1, INPUT_TENSOR_SIZE, INPUT_TENSOR_SIZE]);
    return inputTensor;
}


// prediction du model
function postprocess(logits) {
    let maxScore = -Infinity;
    let prediction = -1;

    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxScore) {
            maxScore = logits[i];
            prediction = i;
        }
    }
    return prediction;
}


canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

canvas.addEventListener('mouseout', () => {
    isDrawing = false;
});

clearButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    predictionResult.textContent = '';
});


predictButton.addEventListener('click', async () => {
    if (!inferenceSession) {
        predictionResult.textContent = 'Model not ready...';
        return;
    }

    predictionResult.textContent = 'Predicting...';

    try {
        const inputTensor = preprocess(canvas);

        const inputName = inferenceSession.inputNames[0];
        const outputName = inferenceSession.outputNames[0];

        const feeds = { [inputName]: inputTensor };
        const results = await inferenceSession.run(feeds);

        const logits = results[outputName].data;
        const prediction = postprocess(logits);

        predictionResult.textContent = prediction;

    } catch (e) {
        console.error("Prediction failed:", e);
        predictionResult.textContent = 'Prediction Error! Check console.';
    }
});