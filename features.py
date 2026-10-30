
import numpy as np
import pandas as pd
from scipy.signal import welch


CHANNELS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
]

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def _extract_channel_features(signal, sfreq, channel_name):
    """Extract statistical features and, for EEG, spectral features."""

    signal = np.asarray(signal, dtype=float)

    features = {
        f"{channel_name}_mean": float(np.mean(signal)),
        f"{channel_name}_std": float(np.std(signal)),
        f"{channel_name}_rms": float(
            np.sqrt(np.mean(signal ** 2))
        ),
        f"{channel_name}_ptp": float(np.ptp(signal)),
    }

    if not channel_name.startswith("EEG"):
        return features

    # Compute the power spectral density ONCE per EEG epoch.
    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=min(len(signal), 512),
    )

    powers = {}

    # Reuse the same PSD for every frequency band.
    for band_name, (low, high) in EEG_BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)

        power = (
            float(np.trapezoid(psd[mask], frequencies[mask]))
            if mask.sum() >= 2
            else 0.0
        )

        powers[band_name] = power
        features[f"{channel_name}_{band_name}_power"] = power

    total_power = sum(powers.values())

    for band_name, power in powers.items():
        features[f"{channel_name}_{band_name}_relative"] = (
            power / total_power if total_power > 0 else 0.0
        )

    return features


def _extract_channel_features(signal, sfreq, channel_name):
    """Extract statistical and spectral features from one epoch."""

    signal = np.asarray(signal, dtype=float)

    features = {
        f"{channel_name}_mean": float(np.mean(signal)),
        f"{channel_name}_std": float(np.std(signal)),
        f"{channel_name}_rms": float(np.sqrt(np.mean(signal ** 2))),
        f"{channel_name}_ptp": float(np.ptp(signal)),
    }

    if channel_name.startswith("EEG"):
        # Calculate the PSD only ONCE for this epoch.
        frequencies, psd = welch(
            signal,
            fs=sfreq,
            nperseg=min(len(signal), 512),
        )

        powers = {}

        # Reuse the same PSD for all five frequency bands.
        for band_name, (low, high) in EEG_BANDS.items():
            mask = (frequencies >= low) & (frequencies < high)

            if mask.sum() < 2:
                power = 0.0
            else:
                power = float(
                    np.trapezoid(psd[mask], frequencies[mask])
                )

            powers[band_name] = power

        total_power = sum(powers.values())

        for band_name, power in powers.items():
            features[f"{channel_name}_{band_name}_power"] = power

            features[f"{channel_name}_{band_name}_relative"] = (
                power / total_power if total_power > 0 else 0.0
            )

    return features


def extract_recording_features(
    raw,
    epoch_table,
    subject_id,
    recording_id,
    batch_size=256,
):
    """
    Vectorized feature extraction in batches.

    Preserves the existing statistical features, Welch settings,
    frequency bands, relative powers, labels and metadata.
    """

    from scipy.signal import welch

    missing = [ch for ch in CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing required channels: {missing}")

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    sfreq = raw.info["sfreq"]
    samples_per_epoch = int(round(30 * sfreq))

    # Read the selected channels once.
    signals = raw.get_data(picks=CHANNELS)

    # Keep only epochs whose sample boundaries are valid.
    starts = np.rint(
        epoch_table["start_sec"].to_numpy() * sfreq
    ).astype(int)

    ends = np.rint(
        epoch_table["end_sec"].to_numpy() * sfreq
    ).astype(int)

    valid = (
        ((ends - starts) == samples_per_epoch)
        & (starts >= 0)
        & (ends <= signals.shape[1])
    )

    metadata = epoch_table.loc[valid].reset_index(drop=True)
    starts = starts[valid]

    output = []

    for batch_start in range(0, len(metadata), batch_size):
        batch_end = min(batch_start + batch_size, len(metadata))

        batch_meta = metadata.iloc[batch_start:batch_end]
        batch_starts = starts[batch_start:batch_end]

        # Shape: (epochs, channels, samples).
        batch = np.stack([
            signals[:, start:start + samples_per_epoch]
            for start in batch_starts
        ])

        batch_rows = {
            "subject_id": [subject_id] * len(batch_meta),
            "recording_id": [recording_id] * len(batch_meta),
            "epoch_index": batch_meta["epoch_index"].to_numpy(),
            "start_sec": batch_meta["start_sec"].to_numpy(),
            "end_sec": batch_meta["end_sec"].to_numpy(),
            "stage": batch_meta["stage"].to_numpy(),
        }

        # Vectorized statistical features.
        for channel_idx, channel_name in enumerate(CHANNELS):
            x = batch[:, channel_idx, :]

            batch_rows[f"{channel_name}_mean"] = x.mean(axis=1)
            batch_rows[f"{channel_name}_std"] = x.std(axis=1)
            batch_rows[f"{channel_name}_rms"] = np.sqrt(
                np.mean(x ** 2, axis=1)
            )
            batch_rows[f"{channel_name}_ptp"] = np.ptp(x, axis=1)

            if not channel_name.startswith("EEG"):
                continue

            # One Welch call for the entire batch and channel.
            frequencies, psd = welch(
                x,
                fs=sfreq,
                nperseg=min(samples_per_epoch, 512),
                axis=-1,
            )

            powers = {}

            for band_name, (low, high) in EEG_BANDS.items():
                mask = (frequencies >= low) & (frequencies < high)

                if mask.sum() >= 2:
                    power = np.trapezoid(
                        psd[:, mask],
                        frequencies[mask],
                        axis=1,
                    )
                else:
                    power = np.zeros(len(batch))

                powers[band_name] = power
                batch_rows[
                    f"{channel_name}_{band_name}_power"
                ] = power

            total_power = np.sum(
                np.stack(list(powers.values()), axis=1),
                axis=1,
            )

            # Spectral power ratios.
            eps = 1e-20

            batch_rows[
                f"{channel_name}_theta_alpha_ratio"
            ] = (
                powers["theta"] /
                (powers["alpha"] + eps)
            )

            batch_rows[
                f"{channel_name}_delta_beta_ratio"
            ] = (
                powers["delta"] /
                (powers["beta"] + eps)
            )

            # Spectral entropy over the selected 0.5-30 Hz bands.
            # Normalize band powers to form a probability distribution.
            probabilities = np.divide(
                np.stack(list(powers.values()), axis=1),
                total_power[:, None],
                out=np.zeros(
                    (len(batch), len(powers)),
                    dtype=float,
                ),
                where=total_power[:, None] > 0,
            )

            entropy = -np.sum(
                np.where(
                    probabilities > 0,
                    probabilities * np.log2(
                        np.maximum(probabilities, eps)
                    ),
                    0.0,
                ),
                axis=1,
            )

            # Normalize entropy to [0, 1].
            max_entropy = np.log2(len(powers))

            batch_rows[
                f"{channel_name}_spectral_entropy"
            ] = entropy / max_entropy

            for band_name, power in powers.items():
                batch_rows[
                    f"{channel_name}_{band_name}_relative"
                ] = np.divide(
                    power,
                    total_power,
                    out=np.zeros_like(power),
                    where=total_power > 0,
                )

        output.append(pd.DataFrame(batch_rows))

    if not output:
        return pd.DataFrame()

    return pd.concat(output, ignore_index=True)

