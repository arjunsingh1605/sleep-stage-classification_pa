
import numpy as np
import pandas as pd


STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


def build_epoch_table(raw, epoch_seconds=30):
    """
    Build metadata for complete, valid, fixed-duration epochs.

    An epoch is retained only when its entire interval is covered
    by a recognized sleep-stage annotation.

    Returns a DataFrame with epoch start/end times and stage labels.
    """

    if epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be positive.")

    annotations = pd.DataFrame({
        "onset": raw.annotations.onset,
        "duration": raw.annotations.duration,
        "description": raw.annotations.description,
    })

    annotations["end"] = (
        annotations["onset"] + annotations["duration"]
    )

    # Zero-duration markers cannot label an epoch.
    annotations = annotations.loc[
        annotations["duration"] > 0
    ].copy()

    # Only recognized sleep-stage annotations can provide labels.
    annotations["stage"] = annotations["description"].map(STAGE_MAP)
    annotations = annotations.loc[
        annotations["stage"].notna()
    ].copy()

    annotations = annotations.sort_values("onset").reset_index(drop=True)

    # Use a grid anchored at the recording start.
    recording_start = 0.0
    recording_end = raw.n_times / raw.info["sfreq"]

    rows = []
    n_epochs = int(
        np.floor((recording_end - recording_start) / epoch_seconds)
    )

    for i in range(n_epochs):
        start = recording_start + i * epoch_seconds
        end = start + epoch_seconds

        # Find annotations that cover this entire epoch.
        covering = annotations.loc[
            (annotations["onset"] <= start + 1e-9)
            & (annotations["end"] >= end - 1e-9)
        ]

        # Require exactly one covering annotation.
        if len(covering) != 1:
            continue

        rows.append({
            "epoch_index": i,
            "start_sec": start,
            "end_sec": end,
            "stage": covering.iloc[0]["stage"],
        })

    return pd.DataFrame(
        rows,
        columns=["epoch_index", "start_sec", "end_sec", "stage"],
    )
