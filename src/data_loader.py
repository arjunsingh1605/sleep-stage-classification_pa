from pathlib import Path
import mne
import pandas as pd

def load_psg(psg_path):
    """
    Load a Sleep-EDF PSG EDF file.

    Parameters
    ----------
    psg_path : str or Path
        Path to the PSG EDF file.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object containing the physiological signals.
    """

    psg_path = Path(psg_path)

    if not psg_path.exists():
        raise FileNotFoundError(f"PSG file not found: {psg_path}")

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=False,
        verbose=False
    )

    return raw


def load_hypnogram(hypnogram_path):
    """
    Load a Sleep-EDF hypnogram annotation EDF file.

    Parameters
    ----------
    hypnogram_path : str or Path
        Path to the hypnogram EDF file.

    Returns
    -------
    annotations : mne.Annotations
        Sleep-stage annotations.
    """

    hypnogram_path = Path(hypnogram_path)

    if not hypnogram_path.exists():
        raise FileNotFoundError(
            f"Hypnogram file not found: {hypnogram_path}"
        )

    annotations = mne.read_annotations(hypnogram_path)

    return annotations


def load_recording(psg_path, hypnogram_path):
    """
    Load a PSG recording and its corresponding hypnogram,
    then attach the annotations to the PSG recording.

    Returns
    -------
    raw : mne.io.Raw
        PSG recording with sleep-stage annotations attached.
    """

    raw = load_psg(psg_path)
    annotations = load_hypnogram(hypnogram_path)

    raw.set_annotations(annotations)

    return raw

    
def analyze_annotations(raw, epoch_seconds=30):
    """
    Explore sleep-stage annotations before constructing epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        PSG recording with hypnogram annotations attached.
    epoch_seconds : int
        Intended epoch duration, in seconds.

    Returns
    -------
    dict
        DataFrames containing annotation details, stage summaries,
        and gaps/overlaps between consecutive annotations.
    """

    annotations = raw.annotations

    # 1. Convert annotations into a tabular representation.
    annotation_df = pd.DataFrame({
        "onset_sec": annotations.onset,
        "duration_sec": annotations.duration,
        "description": annotations.description,
    })

    annotation_df["end_sec"] = (
        annotation_df["onset_sec"]
        + annotation_df["duration_sec"]
    )

    # 2. Check whether each interval's duration is a multiple
    #    of the intended epoch length.
    annotation_df["duration_remainder"] = (
        annotation_df["duration_sec"] % epoch_seconds
    )

    # 3. Check whether each annotation starts on the intended
    #    epoch grid, relative to the recording's first sample.
    annotation_df["onset_remainder"] = (
        annotation_df["onset_sec"] % epoch_seconds
    )

    # 4. Estimate how many complete epochs fit inside each
    #    annotation interval.
    annotation_df["complete_epochs"] = (
        annotation_df["duration_sec"] // epoch_seconds
    ).astype(int)

    # 5. Summarize each annotation category.
    stage_summary = (
        annotation_df
        .groupby("description")
        .agg(
            annotation_count=("description", "size"),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            min_duration_sec=("duration_sec", "min"),
            max_duration_sec=("duration_sec", "max"),
            complete_epochs=("complete_epochs", "sum"),
        )
        .reset_index()
        .sort_values("total_duration_sec", ascending=False)
    )

    stage_summary["total_duration_hours"] = (
        stage_summary["total_duration_sec"] / 3600
    )

    # 6. Find gaps and overlaps between consecutive annotations.
    #    Positive gap = uncovered interval.
    #    Negative gap = overlapping annotations.
    ordered = annotation_df.sort_values("onset_sec").reset_index(
        drop=True
    )

    ordered["next_onset_sec"] = ordered["onset_sec"].shift(-1)

    ordered["gap_sec"] = (
        ordered["next_onset_sec"] - ordered["end_sec"]
    )

    gaps_and_overlaps = ordered.loc[
        ordered["gap_sec"].notna()
        & (ordered["gap_sec"].abs() > 1e-6),
        [
            "onset_sec",
            "end_sec",
            "description",
            "next_onset_sec",
            "gap_sec",
        ],
    ].copy()

    gaps_and_overlaps["interval_type"] = gaps_and_overlaps[
        "gap_sec"
    ].apply(
        lambda value: "GAP" if value > 0 else "OVERLAP"
    )

    # 7. Report annotation alignment checks.
    duration_misaligned = annotation_df.loc[
        annotation_df["duration_remainder"].abs() > 1e-6
    ]

    onset_misaligned = annotation_df.loc[
        annotation_df["onset_remainder"].abs() > 1e-6
    ]

    print("ANNOTATION EDA")
    print("=" * 50)
    print(f"Recording duration: {raw.times[-1]:,.2f} seconds")
    print(f"Annotation count: {len(annotation_df)}")
    print(f"Intended epoch length: {epoch_seconds} seconds")

    print("\nDuration alignment")
    print(
        "Annotations whose durations are not multiples of "
        f"{epoch_seconds}s: {len(duration_misaligned)}"
    )

    print("\nOnset alignment")
    print(
        "Annotations whose starts are not aligned to the "
        f"{epoch_seconds}s grid: {len(onset_misaligned)}"
    )

    print("\nGaps and overlaps")
    print(
        "Gaps:", (gaps_and_overlaps["interval_type"] == "GAP").sum()
    )
    print(
        "Overlaps:",
        (gaps_and_overlaps["interval_type"] == "OVERLAP").sum()
    )

    print("\nStage summary")
    print(stage_summary.to_string(index=False))

    return {
        "annotations": annotation_df,
        "stage_summary": stage_summary,
        "gaps_and_overlaps": gaps_and_overlaps,
        "duration_misaligned": duration_misaligned,
        "onset_misaligned": onset_misaligned,
    }
