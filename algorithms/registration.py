import SimpleITK as sitk
import numpy as np
import nibabel as nib
import ants
import matplotlib.pyplot as plt
import os

#algorithms
from algorithms.segmentation import k_means

def save_image(image, filename, affine_path=""):

    # If exists affine path, use it, if not, use the filename
    if(affine_path == ""):
        imageUploaded = nib.load(os.path.join("images/1", filename))
    else:
        imageUploaded = nib.load(os.path.join("images/1", affine_path))
        

    affine = imageUploaded.affine
    # Create a nibabel image object from the image data
    image = nib.Nifti1Image(image.astype(np.float32), affine=affine)
    # Save the image as a NIfTI file
    output_path = os.path.join("temp_images", filename)
    nib.save(image, output_path)

def registration(fixed_image_path, moving_image_path, segmentated_image_path, type="Rigid"):
    # Read the fixed and moving images
    fixed_image = ants.image_read(os.path.join("temp_images", fixed_image_path))
    moving_image = ants.image_read(os.path.join("temp_images", moving_image_path))

    # Read the segmentated image, if not exists, make it equal to moving image
    if(segmentated_image_path != ""):
        moving_image_segmented = ants.image_read(os.path.join("temp_images", segmentated_image_path))
    else:
        moving_image_segmented = moving_image

    # Perform rigid registration
    transform = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform=type)

    # Apply K-means clustering to the moving image
    # moving_image_nib = nib.load(os.path.join("temp_images", moving_image_path)).get_fdata()
    # moving_image_segmented = k_means(moving_image_nib, 5)
    # save_image(moving_image_segmented, moving_image_path)
    # moving_image_segmented = ants.image_read(os.path.join("temp_images", moving_image_path))

    # Apply the transformation to the moving image
    registered_image = ants.apply_transforms(fixed=fixed_image, moving=moving_image_segmented, transformlist=transform['fwdtransforms'])

    # Convert the registered image to a NumPy array
    registered_array = registered_image.numpy()

    return registered_array

    # # # Save the registered image
    # image_write(registered_image, "registered_image.nii.gz")

def registration_deprecated(fixed_image, moving_image, type="rigid"):

    fixed_image = sitk.ReadImage("images/1/FLAIR.nii.gz")
    moving_image = sitk.ReadImage("images/1/T1.nii.gz")

    # # Convertir las matrices tridimensionales en imágenes SimpleITK
    # fixed_image = sitk.GetImageFromArray(fixed_image.astype(np.float32))
    # moving_image = sitk.GetImageFromArray(moving_image.astype(np.float32))

    # # # Permute the axes of the array to match SimpleITK conventions
    # fixed_image = sitk.PermuteAxes(fixed_image, [2, 1, 0])
    # moving_image = sitk.PermuteAxes(moving_image, [2, 1, 0])

    # Configurar registro
    registration_method = sitk.ImageRegistrationMethod()

    # Configurar metrica de similutud y ponderación
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)



    # Configurar el transformador Rígido
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform)

    # Configurar el optimizador y sus parámetros
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                    numberOfIterations=100, 
                                                    convergenceMinimumValue=1e-6, 
                                                    convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Realiza el registro
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Hacer registro rígido o no rígido dependiendo del parámetro tipo
    if(type == "rigid"):
        # Aplica la transformación rígida a la imagen movida
        registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    else:
        # Aplica la transformación no rígida a la imagen movida
        registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkBSpline, 0.0, moving_image.GetPixelID())


    # Convertir la imagen SimpleITK a una matriz NumPy
    registered_array = sitk.GetArrayFromImage(registered_image)

    # Permutar los ejes de la matriz para que coincida con la convención NIfTI
    registered_array = np.transpose(registered_array, (2, 1, 0))

    return registered_array