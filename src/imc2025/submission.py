from __future__ import annotations

from pathlib import Path

from imc2025.types import Prediction


def array_to_str(array) -> str:
    return ";".join(f"{float(x):.09f}" for x in array)


def none_to_str(n: int) -> str:
    return ";".join(["nan"] * n)


def write_submission(
    predictions_by_dataset: dict[str, list[Prediction]],
    submission_path: str | Path,
    is_train: bool,
) -> Path:
    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    with submission_path.open("w", encoding="utf-8") as fp:
        if is_train:
            fp.write("dataset,scene,image,rotation_matrix,translation_vector\n")
        else:
            fp.write("image_id,dataset,scene,image,rotation_matrix,translation_vector\n")

        for dataset_name, predictions in predictions_by_dataset.items():
            for prediction in predictions:
                cluster_name = (
                    "outliers"
                    if prediction.cluster_index is None
                    else f"cluster{prediction.cluster_index}"
                )
                rotation = (
                    none_to_str(9)
                    if prediction.rotation is None
                    else array_to_str(prediction.rotation.flatten())
                )
                translation = (
                    none_to_str(3)
                    if prediction.translation is None
                    else array_to_str(prediction.translation)
                )
                if is_train:
                    fp.write(
                        f"{dataset_name},{cluster_name},{prediction.filename},{rotation},{translation}\n"
                    )
                else:
                    fp.write(
                        f"{prediction.image_id},{dataset_name},{cluster_name},{prediction.filename},{rotation},{translation}\n"
                    )
    return submission_path
