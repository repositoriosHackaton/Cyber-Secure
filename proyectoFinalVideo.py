import requests
import shutil
import os
from pathlib import Path
import cv2
import numpy as np
import PIL
from typing import Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import facenet_pytorch
from facenet_pytorch import MTCNN
import torch
from facenet_pytorch import InceptionResnetV1
import glob
import platform
from PIL import Image
import logging
import pkg_resources
import sys


def guardar_requirements(output_file="requirements.txt"):
    # Obtener los paquetes importados en el script actual
    imported_packages = set(sys.modules.keys())

    # Obtener una lista de todos los paquetes instalados y sus versiones
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["{}=={}".format(i.key, i.version) for i in installed_packages])

    # Filtrar los paquetes instalados que están siendo usados en este script
    used_packages = [pkg for pkg in installed_packages_list if pkg.split("==")[0] in imported_packages]
    
    for package in used_packages:
        print(package)
    
    # Guardar la lista de paquetes y versiones utilizados en este script en un archivo
    with open(output_file, "w") as f:
        for package in used_packages:
            f.write(package + "\n")

    print(f"Lista de requerimientos guardada en '{output_file}'.")
#______________________________extraer_caras___________________________

def extraer_caras(imagen: Union[PIL.Image.Image, np.ndarray],
                  bboxes: np.ndarray,
                  output_img_size: Union[list, tuple, np.ndarray]=[160, 160]) -> None:
    

    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser np.ndarray, PIL.Image. Recibido {type(imagen)}."
        )
        
    # Recorte de cara
    # --------------------------------------------------------------------------
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen)
        
    if len(bboxes) > 0:
        caras = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cara = imagen[y1:y2, x1:x2]
            # Redimensionamiento del recorte
            cara = Image.fromarray(cara)
            cara = cara.resize(tuple(output_img_size))
            cara = np.array(cara)
            caras.append(cara)
            
    caras = np.stack(caras, axis=0)

    return caras


def detectar_caras(imagen: Union[PIL.Image.Image, np.ndarray],
                   detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                   keep_all: bool        = True,
                   min_face_size: int    = 20,
                   thresholds: list      = [0.6, 0.7, 0.7],
                   device: str           = None,
                   min_confidence: float = 0.5,
                   fix_bbox: bool        = True,
                   verbose               = False)-> np.ndarray:
   
    
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray, PIL.Image`. Recibido {type(imagen)}."
        )

    if detector is None:
        print('Iniciando detector MTCC')
        detector = MTCNN(
                        keep_all      = keep_all,
                        min_face_size = min_face_size,
                        thresholds    = thresholds,
                        post_process  = False,
                        device        = device
                   )
        
    # Detección de caras
    # --------------------------------------------------------------------------
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32)
        
    bboxes, probs = detector.detect(imagen, landmarks=False)
    
    if bboxes is None:
        bboxes = np.array([])
        probs  = np.array([])
    else:
        # Se descartan caras con una probabilidad estimada inferior a `min_confidence`.
        bboxes = bboxes[probs > min_confidence]
        probs  = probs[probs > min_confidence]
        
    print(f'Número total de caras detectadas: {len(bboxes)}')
    print(f'Número final de caras seleccionadas: {len(bboxes)}')

    # Corregir bounding boxes
    #---------------------------------------------------------------------------
    # Si alguna de las esquinas de la bounding box está fuera de la imagen, se
    # corrigen para que no sobrepase los márgenes.
    if len(bboxes) > 0 and fix_bbox:       
        for i, bbox in enumerate(bboxes):
            if bbox[0] < 0:
                bboxes[i][0] = 0
            if bbox[1] < 0:
                bboxes[i][1] = 0
            if bbox[2] > imagen.shape[1]:
                bboxes[i][2] = imagen.shape[1]
            if bbox[3] > imagen.shape[0]:
                bboxes[i][3] = imagen.shape[0]

    # Información de proceso
    # ----------------------------------------------------------------------
    if verbose:
        print("----------------")
        print("Imagen escaneada")
        print("----------------")
        print(f"Caras detectadas: {len(bboxes)}")
        print(f"Correción bounding boxes: {ix_bbox}")
        print(f"Coordenadas bounding boxes: {bboxes}")
        print(f"Confianza bounding boxes:{probs} ")
        print("")
        
    return bboxes.astype(int)


def crear_diccionario_referencias(folder_path:str,
                                  dic_referencia:dict=None,
                                  detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                                  min_face_size: int=40,
                                  thresholds: list=[0.6, 0.7, 0.7],
                                  min_confidence: float=0.9,
                                  encoder=None,
                                  device: str=None,
                                  verbose: bool=False)-> dict:
    
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not os.path.isdir(folder_path):
        raise Exception(
            f"Directorio {folder_path} no existe."
        )
        
    if len(os.listdir(folder_path) ) == 0:
        raise Exception(
            f"Directorio {folder_path} está vacío."
        )
    
    
    if detector is None:
        print('Iniciando detector MTCC')
        detector = MTCNN(
                        keep_all      = False,
                        post_process  = False,
                        min_face_size = min_face_size,
                        thresholds    = thresholds,
                        device        = device
                   )
    
    if encoder is None:
        print('Iniciando encoder InceptionResnetV1')
        encoder = InceptionResnetV1(
                        pretrained = 'vggface2',
                        classify   = False,
                        device     = device
                   ).eval()
        
    
    new_dic_referencia = {}
    folders = glob.glob(folder_path + "/*")
    
    for folder in folders:
        
        if platform.system() in ['Linux', 'Darwin']:
            identidad = folder.split("/")[-1]
        else:
            identidad = folder.split("\\")[-1]
                                     
        print(f'Obteniendo embeddings de: {identidad}')
        embeddings = []
        # Se lista todas las imagenes .jpg .jpeg .tif .png
        path_imagenes = glob.glob(folder + "/*.jpg")
        path_imagenes.extend(glob.glob(folder + "/*.jpeg"))
        path_imagenes.extend(glob.glob(folder + "/*.tif"))
        path_imagenes.extend(glob.glob(folder + "/*.png"))
        print(f'Total imagenes referencia: {len(path_imagenes)}')
        
        for path_imagen in path_imagenes:
            print(f'Leyendo imagen: {path_imagen}')
            imagen = Image.open(path_imagen)
            # Si la imagen es RGBA se pasa a RGB
            if np.array(imagen).shape[2] == 4:
                imagen  = np.array(imagen)[:, :, :3]
                imagen  = Image.fromarray(imagen)
                
            bbox = detectar_caras(
                        imagen,
                        detector       = detector,
                        min_confidence = min_confidence,
                        verbose        = False
                    )
            
            if len(bbox) > 1:
                print(
                    f'Más de 2 caras detectadas en la imagen: {path_imagen}. '
                    f'Se descarta la imagen del diccionario de referencia.'
                )
                continue
                
            if len(bbox) == 0:
                print(
                    f'No se han detectado caras en la imagen: {path_imagen}.'
                )
                continue
                
            cara = extraer_caras(imagen, bbox)
            embedding = calcular_embeddings(cara, encoder=encoder)
            embeddings.append(embedding)
            
        if verbose:
            print(f"Identidad: {identidad} --- Imágenes referencia: {len(embeddings)}")
            
        embedding_promedio = np.array(embeddings).mean(axis = 0)
        new_dic_referencia[identidad] = embedding_promedio
        print(f"Elementos en new_dic_referencia: {len(new_dic_referencia)}")
        print("_________________________________________")
    if dic_referencia is not None:
        dic_referencia.update(new_dic_referencia)
        return dic_referencia
    else:
        return new_dic_referencia
    
        
def descargar_video(url, nombre_archivo):
    # Crear la carpeta 'videos' si no existe
    carpeta_videos = Path('videos')
    carpeta_videos.mkdir(exist_ok=True)

    # Ruta completa del archivo
    ruta_archivo = carpeta_videos / nombre_archivo
    
    # Realiza la solicitud GET al servidor
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Lanza una excepción si hay un error en la solicitud
        # Abre un archivo local con el nombre especificado para escribir el contenido descargado
        with open(ruta_archivo, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print(f'Video descargado y guardado en {ruta_archivo}')


def pipeline_deteccion_video(path_input_video: str,
                             dic_referencia: dict,
                             path_output_video: str=os.getcwd(),
                             detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                             keep_all: bool=True,
                             min_face_size: int=40,
                             thresholds: list=[0.6, 0.7, 0.7],
                             device: str=None,
                             min_confidence: float=0.5,
                             fix_bbox: bool=True,
                             output_img_size: Union[list, tuple, np.ndarray]=[160, 160],
                             encoder=None,
                             threshold_similaridad: float=0.5,
                             ax=None,
                             verbose=False)-> None:
    

    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not os.path.isfile(path_input_video):
        raise Exception(
            f"El archivo {path_input_video} no existe."
        )
        
        
    capture = cv2.VideoCapture(path_input_video)
    input_frames = []
    output_frames = []

    frame_exist = True
    while(frame_exist):
        frame_exist, frame = capture.read()

        if not frame_exist:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frames.append(frame)
    capture.release()


    for frame in tqdm(input_frames):

        bboxes = detectar_caras(
                        imagen         = frame,
                        detector       = detector,
                        keep_all       = keep_all,
                        min_face_size  = min_face_size,
                        thresholds     = thresholds,
                        device         = device,
                        min_confidence = min_confidence,
                        fix_bbox       = fix_bbox
                      )

        frame_np = np.array(frame)
        
        if len(bboxes) == 0:
            logging.info('No se han detectado caras en la imagen.')
            
            output_frames.append(frame_procesado)
        else:
            #Censurar cara del video
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                w = x2 - x1
                h = y2 - y1
                frame_np = blur_face(frame_np, x1, y1, w, h)
            frame_procesado = frame_np         
            
        output_frames.append(frame_procesado)

    if len(output_frames) > 0:
        frame_size = (output_frames[0].shape[1], output_frames[0].shape[0])
        out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'MP4V'), 25, frame_size)

        for frame in output_frames:
            # Convertir la imagen de RGB a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
            
    return


def mostrarVideo(path_output_video):
    cap = cv2.VideoCapture(path_output_video)

    while cap.isOpened():
        ret, im = cap.read()
        if ret == False:
            break
        cv2.imshow("imagen",im)

        if cv2.waitKey(1) == 27:
            break


def blur_face(image, x, y, w, h, ksize=(51, 51)):
    x = max(0, x)
    y = max(0, y)
    w = min(image.shape[1] - x, w)
    h = min(image.shape[0] - y, h)
    
    # Aquí extraemos la región de la cara y aplicamos el desenfoque gaussiano como censura.
    face = image[y:y+h, x:x+w]
    face = cv2.GaussianBlur(face, ksize, 0)
    image[y:y+h, x:x+w] = face
    return image

# URL del video que deseas descargar
url_video = 'https://github.com/RubenDHerrera/Proyecto-Final-Samsung-Innovation-Campus/raw/c67efbcabcb943affaffb5e564aef6b7d607f2c2/videoplayback.mp4'
#url_video = 'https://github.com/RubenDHerrera/Proyecto-Final-Samsung-Innovation-Campus/raw/c67efbcabcb943affaffb5e564aef6b7d607f2c2/video_modern_family.mp4'
#el lionk se consigue al darle click derecho ael botoncito raw y copiar el enlace
# Nombre del archivo donde se guardará el video
nombre_archivo = 'videoplayback.mp4'

descargar_video(url_video, nombre_archivo)

# Detectar si se dispone de GPU cuda
# ==============================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(F'Running on device: {device}')

# Crear diccionario de referencia para cada persona
# ==============================================================================
dic_referencias = crear_diccionario_referencias(
                    folder_path    = './videos',
                    min_face_size  = 40,
                    min_confidence = 0.9,
                    device         = device,
                    verbose        = True
                  )


pipeline_deteccion_video(
    path_input_video      = 'videos/videoplayback.mp4',
    path_output_video     = 'videos/video_processed.mp4',
    dic_referencia        = dic_referencias,
    threshold_similaridad = 0.4,
)

path_output_video  = 'videos/video_processed.mp4'
mostrarVideo(path_output_video)

# Ejecutar la función para guardar los requerimientos en requirements.txt
guardar_requirements("requirements.txt")