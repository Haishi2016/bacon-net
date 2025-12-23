# EEG Eye State Classification

This sample demonstrates using BACON for classifying eye state (open/closed) from EEG sensor readings.

## Dataset

- **Source**: UCI Machine Learning Repository (ID: 264)
- **Full Dataset**: 14,980 EEG measurements (117 seconds of continuous recording)
- **Sampled Dataset**: 5,000 instances (for faster training)
- **Features**: 14 EEG sensor readings (continuous values)
- **Target**: `eyeDetection` (0 = eyes open, 1 = eyes closed)
- **Task**: Binary classification

## Features

The dataset contains readings from 14 EEG electrodes positioned around the scalp using the Emotiv EEG Neuroheadset:

### Left Hemisphere (7 sensors)
- **AF3**: Anterior frontal left
- **F7**: Frontal left
- **F3**: Frontal left-center
- **FC5**: Frontal-central left
- **T7**: Temporal left
- **P7**: Parietal left
- **O1**: Occipital left

### Right Hemisphere (7 sensors)
- **AF4**: Anterior frontal right
- **F8**: Frontal right
- **F4**: Frontal right-center
- **FC6**: Frontal-central right
- **T8**: Temporal right
- **P8**: Parietal right
- **O2**: Occipital right

## Model Configuration

The model uses standard parameters adapted for EEG time-series data:

- **Scaler**: `SigmoidScaler(alpha=3, beta=-1)` for normalizing EEG signals
- **Weight mode**: Trainable (allows learning optimal sensor weights)
- **Training attempts**: 10
- **Hierarchical group size**: 8 (groups sensors for structured learning)
- **Epochs**: 3000 per attempt
- **Test split**: 20%
- **Acceptance threshold**: 90%
- **Sampling**: 5000 instances from 14,980 (for efficiency)

## Expected Results

EEG eye state classification is a well-studied problem with high accuracy:

- **Test Accuracy**: 85-95%
- **Important Sensors**: Likely occipital (O1, O2) and frontal (AF3, AF4, F3, F4) regions
- **Temporal patterns**: Sequential nature of data may affect results

## Running the Sample

```bash
cd samples/eeg-eye-state
python main.py
```

Compare with baselines:
```bash
python main-lr.py
python main-rf.py  
python main-xgb.py
```

## Key Findings

The BACON model can identify:

1. **Critical EEG sensors**: Which electrodes provide most discriminative information
2. **Spatial patterns**: Left vs right hemisphere contributions
3. **Regional importance**: Frontal, temporal, parietal, or occipital dominance
4. **Sensor redundancy**: Which sensors provide overlapping information

The interpretable tree structure reveals the logical combinations of sensor readings that distinguish eye states.

## Neuroscience Relevance

Understanding EEG patterns for eye state detection has applications in:
- **Brain-computer interfaces (BCI)**: Gaze-independent control systems
- **Sleep monitoring**: Detecting REM vs non-REM sleep stages
- **Attention detection**: Monitoring alertness and cognitive load
- **Medical diagnostics**: Identifying abnormal eye movement patterns
- **Drowsiness detection**: Safety applications for drivers/operators

## Technical Notes

- **Temporal structure**: Data is sequential (chronological order), but we treat it as i.i.d. for this task
- **Sampling strategy**: Random sampling from full dataset - temporal correlations are not preserved
- **Data imbalance**: Check class distribution (eyes open vs closed proportion)
- **Sensor symmetry**: Left/right hemisphere sensors may show symmetric patterns
- **Signal preprocessing**: Original signals are continuous; SigmoidScaler normalizes to [0,1] range

## Advanced Considerations

For production systems, consider:
- **Full dataset**: Use all 14,980 instances for maximum accuracy
- **Temporal modeling**: Use sequence models (LSTM, Transformers) to capture temporal dependencies
- **Cross-subject validation**: Test on different individuals
- **Real-time inference**: Optimize model for low-latency prediction
- **Sensor dropout**: Test robustness when some sensors fail

## Citation

Roesler, O. (2013). EEG Eye State. UCI Machine Learning Repository. https://doi.org/10.24432/C57G7J
