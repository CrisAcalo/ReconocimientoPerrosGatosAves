// Variables globales
let modelo;
const clases = ['Perro', 'Gato', 'Ave'];
const tamano = 400;
let camaraActual = "user"; // "user" (frontal) o "environment" (trasera)
let streamActual = null;

// Elementos del DOM
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultadoElemento = document.getElementById("resultado");
const botonCambiarCamara = document.getElementById("cambiarCamara");

// Cargar el modelo TensorFlow.js
async function cargarModelo() {
    modelo = await tf.loadLayersModel("./cnn_AD_PerrosGatosAves/model.json");
    console.log("Modelo cargado correctamente.");
}

// Inicializar la cámara
async function inicializarCamara() {
    if (streamActual) {
        streamActual.getTracks().forEach(track => track.stop()); // Detener cámara anterior
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: tamano, 
                height: tamano, 
                facingMode: camaraActual 
            }
        });

        streamActual = stream; // Guardar referencia al nuevo stream
        video.srcObject = stream;
        video.play();
    } catch (error) {
        alert("No se pudo acceder a la cámara.");
        console.error(error);
    }
}

// Preprocesar la imagen para que sea compatible con el modelo
function preprocesarImagen(imagen) {
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(imagen)
            .resizeNearestNeighbor([100, 100]) 
            .mean(2)                        
            .expandDims(2)                    
            .expandDims()                     
            .toFloat()                         
            .div(255.0);                      
        return tensor;
    });
}

// Realizar la predicción
async function predecir() {
    if (modelo) {
        ctx.drawImage(video, 0, 0, tamano, tamano);

        const resultados = tf.tidy(() => {
            const tensor = preprocesarImagen(canvas);
            const prediccion = modelo.predict(tensor);
            return prediccion.dataSync();
        });

        const indiceMax = resultados.indexOf(Math.max(...resultados));
        const animalReconocido = clases[indiceMax];

        resultadoElemento.textContent = `Animal reconocido: ${animalReconocido}`;
    }

    requestAnimationFrame(predecir);
}

// Cambiar entre la cámara frontal y trasera
botonCambiarCamara.addEventListener("click", async () => {
    camaraActual = camaraActual === "user" ? "environment" : "user";
    await inicializarCamara();
});

// Función principal
async function main() {
    await cargarModelo();
    await inicializarCamara();
    predecir();
}

// Ejecutar la función principal
main();
