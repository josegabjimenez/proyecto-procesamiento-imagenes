import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import SimpleITK as sitk
import os

from algorithms.segmentation import k_means


def remove_brain():
    # Cargar la imagen NIfTI

    nifti_img = nib.load(
        os.path.join("temp_images", "registered_IR.nii.gz")
    )  # Asegúrate de ajustar la ruta y el nombre del archivo

    # Obtener los datos de la imagen
    data = nifti_img.get_fdata()

    # Definir escalas espaciales
    scales = [7.5]  # Escalas para aplicar filtros gaussianos

    # Aplicar filtros gaussianos en diferentes escalas
    filtered_images = []
    for scale in scales:
        # Aplicar filtro gaussiano
        filtered = ndimage.gaussian_filter(data, sigma=scale)
        filtered = k_means(filtered, 2)
        # Crear una nueva imagen nibabel con el cerebro extraído
        brain_extracted_image = nib.Nifti1Image(
            filtered, affine=nifti_img.affine, dtype=np.int16
        )

        # Guardar la imagen con el cerebro extraído en un nuevo archivo
        nib.save(brain_extracted_image, os.path.join("temp_images", "IR_skull.nii.gz"))
        filtered_images.append(filtered)

    # RESTAR UNA IMAGEN

    # Cargar las imágenes
    imagen_original = sitk.ReadImage(
        os.path.join("temp_images", "registered_T1.nii.gz")
    )
    imagen_referencia = sitk.ReadImage(os.path.join("temp_images", "IR_skull.nii.gz"))

    # Modify the metadata of image2 to match image1
    imagen_referencia.SetOrigin(imagen_original.GetOrigin())
    imagen_referencia.SetSpacing(imagen_original.GetSpacing())
    imagen_referencia.SetDirection(imagen_original.GetDirection())

    # Realizar segmentación basada en umbral adaptativo
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mascara_referencia = otsu_filter.Execute(imagen_referencia)

    # Aplicar la máscara a la imagen original
    imagen_sin_craneo = sitk.Mask(imagen_original, mascara_referencia)

    # Obtener los datos de la imagen sin el cráneo
    # Obtener los datos de la imagen sin el cráneo
    data_sin_craneo = sitk.GetArrayFromImage(imagen_sin_craneo)

    # Obtener los datos de la máscara
    data_mascara = sitk.GetArrayFromImage(mascara_referencia)

    # Crear una máscara booleana para los valores cero dentro del cerebro
    mascara_cero_cerebro = (data_sin_craneo == 0) & (data_mascara != 0)

    # Asignar un valor distinto a los valores cero dentro del cerebro
    valor_distinto = 4
    data_sin_craneo[mascara_cero_cerebro] = valor_distinto

    # Crear una nueva imagen SimpleITK con los datos modificados
    imagen_sin_craneo_modificada = sitk.GetImageFromArray(data_sin_craneo)
    imagen_sin_craneo_modificada.CopyInformation(imagen_sin_craneo)

    # Guardar la imagen sin el cráneo

    sitk.WriteImage(
        imagen_sin_craneo_modificada, os.path.join("temp_images", "FLAIR_skull.nii.gz")
    )

    # ----------------------------------------------------------------------------------
    # Quitar cráneo a FLAIR Original
    # ----------------------------------------------------------------------------------
    # Cargar las imágenes

    imagen_original = sitk.ReadImage(os.path.join("temp_images", "FLAIR.nii.gz"))
    imagen_referencia = sitk.ReadImage(os.path.join("temp_images", "IR_skull.nii.gz"))

    # Realizar segmentación basada en umbral adaptativo
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mascara_referencia = otsu_filter.Execute(imagen_referencia)

    # Aplicar la máscara a la imagen original
    imagen_sin_craneo = sitk.Mask(imagen_original, mascara_referencia)

    # Guardar la imagen sin el cráneo

    sitk.WriteImage(
        imagen_sin_craneo,
        os.path.join("temp_images", "FLAIR_original_sin_craneo.nii.gz"),
    )

    # ----------------------------------------------------------------------------------
    # Segmentar lesiones
    # ----------------------------------------------------------------------------------

    image = nib.load(os.path.join("temp_images", "FLAIR_skull.nii.gz"))
    image_data = image.get_fdata()
    image_data_flair_without_skull = nib.load(
        os.path.join("temp_images", "FLAIR_original_sin_craneo.nii.gz")
    ).get_fdata()

    image_data_flair_segmented = k_means(image_data_flair_without_skull, 15, 15)

    # Where the values are 3, replace them in the image_data with a value of 3
    image_data_flair_segmented[:,:,:13] = 0
    image_data = np.where(image_data_flair_segmented == 7, 3, image_data)

    affine = image.affine
    # Create a nibabel image object from the image data
    image = nib.Nifti1Image(image_data.astype(np.float32), affine=affine)
    # Save the image as a NIfTI file
    output_path = os.path.join("temp_images", "FLAIR_skull_lesion.nii.gz")
    nib.save(image, output_path)

    return image_data
