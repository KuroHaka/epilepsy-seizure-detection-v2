import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plotter import process_eeg

st.write("""
# Automatic seizure detection
""")
option = st.selectbox(
    'Select edf file',
        ('chb01_03',
        'chb01_04',
        'chb01_15',
        'chb01_16',
        'chb01_18',
        'chb01_21',
        'chb01_26',
        'chb02_16',
        'chb02_17',
        'chb02_18',
        'chb03_01',
        'chb03_02',
        'chb03_03',
        'chb03_04',
        'chb03_34',
        'chb03_35',
        'chb03_36',
        'chb04_05',
        'chb04_08',
        'chb04_28',
        'chb05_06',
        'chb05_13',
        'chb05_16',
        'chb05_17',
        'chb05_22',
        'chb06_01',
        'chb06_04',
        'chb06_09',
        'chb06_10',
        'chb06_13',
        'chb06_18',
        'chb06_24',
        'chb07_12',
        'chb07_13',
        'chb07_19',
        'chb08_02',
        'chb08_05',
        'chb08_11',
        'chb08_13',
        'chb08_21',
        'chb09_06',
        'chb09_08',
        'chb09_19',
        'chb10_12',
        'chb10_20',
        'chb10_27',
        'chb10_30',
        'chb10_31',
        'chb10_38',
        'chb10_89',
        'chb11_82',
        'chb11_92',
        'chb11_99',
        'chb12_06',
        'chb12_08',
        'chb12_09',
        'chb12_10',
        'chb12_11',
        'chb12_23',
        'chb12_27',
        'chb12_28',
        'chb12_29',
        'chb12_33',
        'chb12_36',
        'chb12_38',
        'chb12_42',
        'chb13_19',
        'chb13_21',
        'chb13_40',
        'chb13_55',
        'chb13_58',
        'chb13_59',
        'chb13_60',
        'chb13_62',
        'chb14_03',
        'chb14_04',
        'chb14_06',
        'chb14_11',
        'chb14_17',
        'chb14_18',
        'chb14_27',
        'chb15_06',
        'chb15_10',
        'chb15_15',
        'chb15_17',
        'chb15_20',
        'chb15_22',
        'chb15_28',
        'chb15_31',
        'chb15_40',
        'chb15_46',
        'chb15_49',
        'chb15_52',
        'chb15_54',
        'chb15_62',
        'chb16_10',
        'chb16_11',
        'chb16_14',
        'chb16_16',
        'chb16_17',
        'chb16_18',
        'chb17a_03',
        'chb17a_04',
        'chb17b_63',
        'chb18_29',
        'chb18_30',
        'chb18_31',
        'chb18_32',
        'chb18_35',
        'chb18_36',
        'chb19_28',
        'chb19_29',
        'chb19_30',
        'chb20_12',
        'chb20_13',
        'chb20_14',
        'chb20_15',
        'chb20_16',
        'chb20_68',
        'chb21_19',
        'chb21_20',
        'chb21_21',
        'chb21_22',
        'chb22_20',
        'chb22_25',
        'chb22_38',
        'chb23_06',
        'chb23_08',
        'chb23_09',
        'chb24_01',
        'chb24_03',
        'chb24_04',
        'chb24_06',
        'chb24_07',
        'chb24_09',
        'chb24_11',
        'chb24_13',
        'chb24_14',
        'chb24_15',
        'chb24_17',
        'chb24_21'))

model = st.selectbox(
    'Select your model',
        ('SVM', 'RF', 'KNN', 'FFNN', 'CNN+FFNN'))

threshold = -1
if model == 'FFNN' or model == 'CNN+FFNN':
    threshold = st.slider('Threshold', -0.5, 1.5, 0.5, 0.01)

if st.button('Autodetect'):
    with st.spinner('Processing EEG data...'):
        result = process_eeg(option, model, str(threshold))

    raw_data = result['raw_data']
    sfreq = result['sfreq']
    ch_names = result['ch_names']
    seizure_start = result['seizure_start']
    seizure_duration = result['seizure_duration']
    detected = result['detected']

    n_channels = len(ch_names)
    total_samples = raw_data.shape[1]
    times = np.arange(total_samples) / sfreq

    window_duration = 50
    view_start = max(0, seizure_start[0] - 5)
    view_end = view_start + window_duration
    start_idx = int(view_start * sfreq)
    end_idx = min(int(view_end * sfreq), total_samples)
    t = times[start_idx:end_idx]
    data_window = raw_data[:, start_idx:end_idx]

    scale = 250e-6
    offsets = np.arange(n_channels) * scale

    # Ground truth plot
    fig_gt, ax_gt = plt.subplots(figsize=(14, 8))
    for i in range(n_channels):
        ax_gt.plot(t, data_window[i] + offsets[i], color='green', linewidth=0.4)
    for onset, dur in zip(seizure_start, seizure_duration):
        ax_gt.axvspan(onset, onset + dur, alpha=0.25, color='orange', label='Ground truth')
    ax_gt.set_yticks(offsets)
    ax_gt.set_yticklabels(ch_names, fontsize=7)
    ax_gt.set_xlabel('Time (s)')
    ax_gt.set_title('EEG — Ground Truth Seizure')
    ax_gt.set_xlim(view_start, view_end)
    handles, labels = ax_gt.get_legend_handles_labels()
    if handles:
        ax_gt.legend([handles[0]], [labels[0]])
    plt.tight_layout()

    st.subheader('Ground Truth')
    st.pyplot(fig_gt)
    plt.close(fig_gt)

    # Prediction plot
    fig_pred, ax_pred = plt.subplots(figsize=(14, 8))
    for i in range(n_channels):
        ax_pred.plot(t, data_window[i] + offsets[i], color='steelblue', linewidth=0.4)
    for onset, dur in zip(seizure_start, seizure_duration):
        ax_pred.axvspan(onset, onset + dur, alpha=0.15, color='orange', label='Ground truth')
    for epoch_idx in detected:
        epoch_start = epoch_idx * 1
        epoch_end = epoch_start + 2
        if epoch_start >= view_start and epoch_start <= view_end:
            ax_pred.axvspan(epoch_start, epoch_end, alpha=0.3, color='red', label='Predicted')
    ax_pred.set_yticks(offsets)
    ax_pred.set_yticklabels(ch_names, fontsize=7)
    ax_pred.set_xlabel('Time (s)')
    ax_pred.set_title('EEG — Model Prediction')
    ax_pred.set_xlim(view_start, view_end)
    handles, labels = ax_pred.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_pred.legend(by_label.values(), by_label.keys())
    plt.tight_layout()

    st.subheader('Prediction')
    st.pyplot(fig_pred)
    plt.close(fig_pred)

    st.success(f'Detected {len(detected)} seizure epochs')
