import sys
import os
# Agregar el directorio base del proyecto (dos niveles arriba) al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join, subdirs
import shutil
import random
import json
import argparse
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

def gather_cases_kits19(kits19_dir):
    """
    Recolecta los casos de KiTS19. Se espera que en kits19_dir existan carpetas con nombre "case_XXXXX"
    que contengan los archivos imaging.nii.gz y segmentation.nii.gz.
    """
    cases = subdirs(kits19_dir, prefix='case_', join=True)
    case_list = []
    for case in cases:
        imaging_file = join(case, 'imaging.nii.gz')
        segmentation_file = join(case, 'segmentation.nii.gz')
        if os.path.exists(imaging_file) and os.path.exists(segmentation_file):
            case_list.append({
                'source': 'kits19',
                'case_path': case,
                'imaging': imaging_file,
                'segmentation': segmentation_file
            })
        else:
            print("Advertencia: Archivos no encontrados en", case)
    return case_list

def gather_cases_kits21(kits21_dir):
    """
    Recolecta los casos de KiTS21. Se espera que en kits21_dir existan carpetas "case_XXXXX" 
    que contengan imaging.nii.gz y las segmentaciones.
    Se utiliza 'aggregated_MAJ_seg.nii.gz' para la segmentación.
    """
    cases = subdirs(kits21_dir, prefix='case_', join=True)
    case_list = []
    for case in cases:
        imaging_file = join(case, 'imaging.nii.gz')
        segmentation_file = join(case, 'aggregated_MAJ_seg.nii.gz')
        if os.path.exists(imaging_file) and os.path.exists(segmentation_file):
            case_list.append({
                'source': 'kits21',
                'case_path': case,
                'imaging': imaging_file,
                'segmentation': segmentation_file
            })
        else:
            print("Advertencia: Archivos no encontrados en", case)
    return case_list

def gather_cases_kits23(kits23_dir):
    """
    Recolecta los casos de KiTS23. Se espera que en kits23_dir existan carpetas "case_XXXXX" 
    que contengan imaging.nii.gz y segmentation.nii.gz. Se busca en la carpeta principal o en 'raw'.
    """
    cases = subdirs(kits23_dir, prefix='case_', join=True)
    case_list = []
    for case in cases:
        imaging_file = join(case, 'imaging.nii.gz')
        if not os.path.exists(imaging_file):
            imaging_file = join(case, 'raw', 'imaging.nii.gz')
        segmentation_file = join(case, 'segmentation.nii.gz')
        if not os.path.exists(segmentation_file):
            segmentation_file = join(case, 'raw', 'segmentation.nii.gz')
        if os.path.exists(imaging_file) and os.path.exists(segmentation_file):
            case_list.append({
                'source': 'kits23',
                'case_path': case,
                'imaging': imaging_file,
                'segmentation': segmentation_file
            })
        else:
            print("Advertencia: Archivos no encontrados en", case)
    return case_list

def convert_kits_unified(kits19_dir: str, kits21_dir: str, kits23_dir: str, nnunet_dataset_id: int = 1):
    """
    Combina los tres datasets y los divide aleatoriamente en:
      - Train: 60%
      - Validation: 20%
      - Test: 20%
      
    Los casos de train y validation se copiarán en imagesTr y labelsTr, y los de test en imagesTs (solo imágenes).
    Se renombra uniformemente cada caso (case_00000, case_00001, ...).
    Además, se genera un archivo splits.json con la información de la división.
    """
    task_name = "Kits"
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)
    out_base = join(nnUNet_raw, foldername)
    imagesTr_dir = join(out_base, "imagesTr")
    labelsTr_dir = join(out_base, "labelsTr")
    imagesTs_dir = join(out_base, "imagesTs")
    maybe_mkdir_p(imagesTr_dir)
    maybe_mkdir_p(labelsTr_dir)
    maybe_mkdir_p(imagesTs_dir)

    # Recolectamos los casos de cada dataset
    cases_kits19 = gather_cases_kits19(kits19_dir)
    cases_kits21 = gather_cases_kits21(kits21_dir)
    cases_kits23 = gather_cases_kits23(kits23_dir)
    
    all_cases = cases_kits19 + cases_kits21 + cases_kits23
    print(f"Total de casos encontrados: {len(all_cases)}")

    # División aleatoria en train (60%), validation (20%) y test (20%)
    random.seed(42)
    random.shuffle(all_cases)
    n_total = len(all_cases)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    print(f"División: {n_train} train, {n_val} validation, {n_test} test")

    # Los casos de entrenamiento (train y validation) se colocan en imagesTr/labelsTr
    train_val_cases = all_cases[:n_train + n_val]
    test_cases = all_cases[n_train + n_val:]

    # Guardamos la información de la división
    split_info = {}

    case_counter = 0
    # Procesamos los casos de train y validation
    for case in train_val_cases:
        new_case_id = f"case_{case_counter:05d}"
        dest_imaging = join(imagesTr_dir, f"{new_case_id}_0000.nii.gz")
        dest_segmentation = join(labelsTr_dir, f"{new_case_id}.nii.gz")
        shutil.copy(case['imaging'], dest_imaging)
        shutil.copy(case['segmentation'], dest_segmentation)
        # Asignamos 'train' a los primeros n_train y 'validation' al resto
        if case_counter < n_train:
            split_info[new_case_id] = "train"
        else:
            split_info[new_case_id] = "validation"
        case_counter += 1

    # Procesamos los casos de test (solo imágenes)
    for case in test_cases:
        new_case_id = f"case_{case_counter:05d}"
        dest_imaging = join(imagesTs_dir, f"{new_case_id}_0000.nii.gz")
        shutil.copy(case['imaging'], dest_imaging)
        split_info[new_case_id] = "test"
        case_counter += 1

    # Guardamos la información de la división en splits.json
    split_file = join(out_base, "splits.json")
    with open(split_file, "w") as f:
        json.dump(split_info, f, indent=4)
    print("Información de la división guardada en:", split_file)

    # Generamos el dataset.json para nnU-Net.
    num_training_cases = len(train_val_cases)
    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "kidney": (1, 2, 3),
                              "masses": (2, 3),
                              "tumor": 2
                          },
                          regions_class_order=(1, 3, 2),
                          num_training_cases=num_training_cases, file_ending='.nii.gz',
                          dataset_name=task_name, reference='none',
                          release='0.1.3',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="Dataset unificado de KiTS19, KiTS21 y KiTS23")
    print("dataset.json generado en:", out_base)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kits19', type=str, default=r"C:\Users\Asus\Documents\TFG\Data\kists19\data",
                        help="Ruta a la carpeta base del dataset KiTS19 (debe contener las carpetas case_XXXXX)")
    parser.add_argument('--kits21', type=str, default=r"C:\Users\Asus\Documents\TFG\Data\kists21\data",
                        help="Ruta a la carpeta base del dataset KiTS21 (normalmente la carpeta 'data' con los case_XXXXX)")
    parser.add_argument('--kits23', type=str, default=r"C:\Users\Asus\Documents\TFG\Data\kists23\dataset",
                        help="Ruta a la carpeta base del dataset KiTS23 (normalmente la carpeta 'dataset' con los case_XXXXX)")
    parser.add_argument('-d', type=int, default=1,
                        help="ID del dataset nnU-Net, por defecto: 1 (para Dataset001_Kits)")
    args = parser.parse_args()

    convert_kits_unified(args.kits19, args.kits21, args.kits23, args.d)
    print("Conversión completada.")
    # Para ejecutar el script, puedes usar el siguiente comando:
    # python C:\Users\Asus\Documents\TFG\Code\nnUNet\nnunetv2\dataset_conversion\Dataset_KiTS_Challenge.py --kits19 "C:\Users\Asus\Documents\TFG\Data\kits19\data" --kits21 "C:\Users\Asus\Documents\TFG\Data\kits21\kits21\data" --kits23 "C:\Users\Asus\Documents\TFG\Data\kits23\dataset" -d 1

