// Variables globales
let modelo;
const clases = ['Perro', 'Gato', 'Ave'];
const tamano = 400;

// Elementos del DOM
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultadoElemento = document.getElementById("resultado");

// Cargar el modelo TensorFlow.js
async function cargarModelo() {
    modelo = await tf.loadLayersModel("./cnn_AD_PerrosGatosAves/model.json");
    console.log("Modelo cargado correctamente.");
}

// Inicializar la cámara
async function inicializarCamara() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: tamano, height: tamano }
        });
        video.srcObject = stream;
        video.play();
    } else {
        alert("Tu navegador no soporta el uso de la cámara.");
    }
}

// Preprocesar la imagen para que sea compatible con el modelo
function preprocesarImagen(imagen) {
    return tf.tidy(() => {
        // Convertir la imagen a un tensor
        const tensor = tf.browser.fromPixels(imagen)
            .resizeNearestNeighbor([100, 100])  // Redimensionar a 100x100
            .mean(2)                            // Convertir a escala de grises
            .expandDims(2)                      // Añadir dimensión del canal (100x100x1)
            .expandDims()                       // Añadir dimensión del batch (1x100x100x1)
            .toFloat()                          // Convertir a float
            .div(255.0);                        // Normalizar (0-1)
        return tensor;
    });
}

// Realizar la predicción
async function predecir() {
    if (modelo) {
        // Capturar la imagen del video
        ctx.drawImage(video, 0, 0, tamano, tamano);

        // Preprocesar la imagen y realizar la predicción dentro de tf.tidy
        const resultados = tf.tidy(() => {
            const tensor = preprocesarImagen(canvas);
            const prediccion = modelo.predict(tensor);
            return prediccion.dataSync();  // Obtener los resultados como un array
        });

        // Obtener la clase con la probabilidad más alta
        const indiceMax = resultados.indexOf(Math.max(...resultados));
        const animalReconocido = clases[indiceMax];

        // Mostrar el resultado en la página
        resultadoElemento.textContent = `Animal reconocido: ${animalReconocido}`;

        // Imprimir los porcentajes en la consola
        console.log('Porcentajes:');
        resultados.forEach((probabilidad, index) => {
            console.log(`${clases[index]}: ${(probabilidad * 100).toFixed(2)}%`);
        });
    }

    // Llamar a la función de predicción nuevamente
    requestAnimationFrame(predecir);
}

// Función principal
async function main() {
    await cargarModelo();
    await inicializarCamara();
    predecir();  // Iniciar el bucle de predicción
}

// Ejecutar la función principal
main();