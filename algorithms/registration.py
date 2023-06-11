import SimpleITK as sitk
import numpy as np
import nibabel as nib

def registration(fixed_image, moving_image, type="rigid"):

    # # Convertir las matrices tridimensionales en imágenes SimpleITK
    fixed_image = sitk.GetImageFromArray(fixed_image.astype(np.float32))
    moving_image = sitk.GetImageFromArray(moving_image.astype(np.float32))

    # # Permute the axes of the array to match SimpleITK conventions
    fixed_image = sitk.PermuteAxes(fixed_image, [2, 1, 0])
    moving_image = sitk.PermuteAxes(moving_image, [2, 1, 0])

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