
from dataclasses import dataclass
from typing import Any
import torch
from tqdm import tqdm
from vibromodes.hdf5_dataset import BatchData
from vibromodes.kirchhoff import tr_velocity_field_to_frequency_response
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

from vibromodes.velocity_field import field_dict2frf


def compute_peak_distances(actual_amplitudes, predicted_amplitudes, actual_frequencies, predicted_frequencies):
    # Compute amplitude and frequency differences using broadcasting
    amplitude_diffs = np.abs(actual_amplitudes[:, None] - predicted_amplitudes)
    frequency_diffs = np.abs(actual_frequencies[:, None] - predicted_frequencies)

    # Compute distance matrix using given weights
    distance_matrix = frequency_diffs

    return distance_matrix, amplitude_diffs, frequency_diffs



def _peak_frequency_error(actual_amplitudes, predicted_amplitudes, prominence_threshold=0.05):
    # Find peaks
    actual_peaks, _ = find_peaks(actual_amplitudes, prominence=prominence_threshold, wlen=100)
    predicted_peaks, _ = find_peaks(predicted_amplitudes, prominence=prominence_threshold, wlen=100)

    # Get peak amplitudes
    actual_peak_amplitudes = actual_amplitudes[actual_peaks]
    predicted_peak_amplitudes = predicted_amplitudes[predicted_peaks]
    # Compute distance matrix
    distance_matrix, amplitude_diffs, frequency_diffs = compute_peak_distances(actual_peak_amplitudes, predicted_peak_amplitudes, actual_peaks,
                                                                               predicted_peaks)

    # Perform optimal assignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Compute mean distance for matched peaks
    matched_amplitude_distance = np.mean(amplitude_diffs[row_indices, col_indices])
    matched_frequency_distance = np.mean(frequency_diffs[row_indices, col_indices])
    # Compute number of non-matched peaks
    peak_ratio = np.abs(len(predicted_peaks)) / len(actual_peaks)
    save_peak_ratio = np.min((peak_ratio, len(actual_peaks) / np.abs(len(predicted_peaks))))
    return peak_ratio, matched_amplitude_distance, matched_frequency_distance, len(actual_peaks), save_peak_ratio



def peak_frequency_error(actual_amplitudes, predicted_amplitudes, prominence_threshold=0.5):
    # Number of samples
    n_samples = actual_amplitudes.shape[0]
    ratios, save_ratios, n_peaks = [], [], []
    amplitude_distance, frequency_distance = [], []

    # Loop over samples
    for i in range(n_samples):
        peak_ratio, matched_amplitude_distance, matched_frequency_distance, n_peak, save_peak_ratio = _peak_frequency_error(
            actual_amplitudes[i], predicted_amplitudes[i], prominence_threshold
        )
        ratios.append(peak_ratio), n_peaks.append(n_peak), save_ratios.append(save_peak_ratio)
        amplitude_distance.append(matched_amplitude_distance), frequency_distance.append(matched_frequency_distance)
    save_rmean = 1 - np.nanmean(save_ratios)

    results = {"save_rmean": save_rmean, "amplitude_distance": np.nanmean(amplitude_distance),
                 "frequency_distance": np.nanmean(frequency_distance)}

    return results


def calc_metrics(pred_frfs,tgt_frfs):


    mse_loss = torch.nn.functional.mse_loss(pred_frfs,tgt_frfs).item()

    result = {"mse": mse_loss}


    #if we have enough frequencies we can calcualte the peak_frequency erros

    if(pred_frfs.shape[1]>150):
        result |= peak_frequency_error(pred_frfs.detach().cpu().numpy(),tgt_frfs.detach().cpu().numpy())
    return result



@dataclass
class FRFResults:
    target_frfs : Any = None #numpy
    pred_frfs : Any = None #numpy
    freqs : Any = None #numpy
    ids : Any = None #numpy

    pred_full_frf: Any = None #numpy
    pred_eigenfreqs: Any = None #list of numpy array





    

@torch.no_grad()
def generate_frfs(model,dataloader,device,ret_full_frf=False,tqdm_enable=False) -> FRFResults:
    model.eval()

    tgt_frfs = []
    pred_frfs = []
    ids = []
    all_freqs =[] 
    full_pred_frfs = []
    all_eigenfreqs = []

    full_freqs = torch.linspace(1,300,300).unsqueeze(0).to(device)


    for batch in tqdm(dataloader,disable=not tqdm_enable):
        batch : BatchData = batch.to(device,non_blocking=True)
        pattern = batch.pattern
        freqs = batch.freqs
        phy_para = batch.phy_para
        tgt_z_vel = batch.z_vel

        ids.append(batch.id.detach().cpu())

        pred_field,_ = model(pattern,phy_para.to_dict(),freqs)
        pred_frf = field_dict2frf(pred_field,normalize=True)
        
        if ret_full_frf:
            full_freqs_tmp = full_freqs.repeat([pattern.shape[0],1])
            pred_field,_ = model(pattern,phy_para.to_dict(),full_freqs_tmp)

            full_pred_frf = field_dict2frf(pred_field,normalize=True)
            full_pred_frfs.append(full_pred_frf.detach().cpu())


        tgt_frf = field_dict2frf(tgt_z_vel,normalize=True)
        #pred_frf = tr_velocity_field_to_frequency_response(pred_z_vel,normalization=True)

        tgt_frfs.append(tgt_frf.detach().cpu())
        pred_frfs.append(pred_frf.detach().cpu())
        all_freqs.append(freqs.detach().cpu())


    tgt_frfs = torch.concatenate(tgt_frfs) 
    pred_frfs = torch.concatenate(pred_frfs) 
    ids = torch.concatenate(ids)
    if len(all_freqs)>0:
        all_freqs = torch.concatenate(all_freqs)
    else:
        all_freqs = None
    
    if(len(full_pred_frfs)>0):
        full_pred_frfs = torch.concatenate(full_pred_frfs)
    else:
        full_pred_frfs = None

    return FRFResults(
        target_frfs=tgt_frfs,
        pred_frfs=pred_frfs,
        ids=ids,
        freqs=all_freqs,
        pred_full_frf=full_pred_frfs,
        pred_eigenfreqs=all_eigenfreqs,
    )    

    if not ret_full_frf:
        return pred_frfs,tgt_frfs,ids
    else:
        return pred_frfs,tgt_frfs,ids,all_freqs,full_pred_frfs




def evaluate_model(model,dataloader,device):
    frfs = generate_frfs(model,dataloader,device)
    return calc_metrics(frfs.pred_frfs,frfs.target_frfs)

