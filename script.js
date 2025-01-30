// Variables globales
let modelo;
const clases = ['Perro', 'Gato', 'Ave'];
const tamano = 400;
let camaraActual = "user"; // "user" (frontal) o "environment" (trasera)
let streamActual = null;

// Elementos del DOM
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const otrocanvas = document.createElement("canvas"); // Para preprocesar imágenes
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
        streamActual.getTracks().forEach(track => track.stop());
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: tamano, height: tamano, facingMode: camaraActual }
        });
        streamActual = stream;
        video.srcObject = stream;
        video.play();
    } catch (error) {
        alert("No se pudo acceder a la cámara.");
        console.error(error);
    }
}

// Preprocesar imagen para el modelo
function preprocesarImagen() {
    return tf.tidy(() => {
        otrocanvas.width = 100;
        otrocanvas.height = 100;
        const ctx2 = otrocanvas.getContext("2d");
        ctx2.drawImage(video, 0, 0, 100, 100);
        
        const imgData = ctx2.getImageData(0, 0, 100, 100);
        let arr = [];
        let arr100 = [];

        for (let p = 0; p < imgData.data.length; p += 4) {
            let gris = (imgData.data[p] + imgData.data[p + 1] + imgData.data[p + 2]) / (3 * 255);
            arr100.push([gris]);
            if (arr100.length === 100) {
                arr.push(arr100);
                arr100 = [];
            }
        }
        return tf.tensor4d([arr]);
    });
}

// Realizar la predicción
async function predecir() {
    if (modelo) {
        ctx.drawImage(video, 0, 0, tamano, tamano);
        
        const tensor = preprocesarImagen();
        const resultados = modelo.predict(tensor).dataSync();
        tensor.dispose();
        
        const indiceMax = resultados.indexOf(Math.max(...resultados));
        resultadoElemento.textContent = `Animal reconocido: ${clases[indiceMax]}`;
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