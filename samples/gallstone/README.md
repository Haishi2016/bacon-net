# Gallstone Disease Prediction with BACON

This sample demonstrates using BACON for gallstone disease prediction using bioimpedance and laboratory data.

## Dataset

**Source:** [UCI Machine Learning Repository - Gallstone](https://archive-beta.ics.uci.edu/dataset/1150/gallstone-1)

**Citation:** Esen, I., Arslan, H., Aktürk Esen, S., Gülşen, M., Kültekin, N., & Özdemir, O. (2024). Early prediction of gallstone disease with a machine learning-based method from bioimpedance and laboratory data. *Medicine*, 103(5), e37258. https://doi.org/10.1097/md.0000000000037258

### Dataset Information

- **Instances:** 320 individuals (161 with gallstone disease, 159 without)
- **Features:** 37 clinical measures (demographic, bioimpedance, laboratory)
- **Task:** Binary classification (gallstone disease vs. healthy)
- **Missing Values:** None
- **Class Balance:** Balanced (50.3% positive, 49.7% negative)

### Features

**Demographic (5 features):**
- Age, Gender, Height, Weight, BMI

**Bioimpedance Data (8 features):**
- Total body water (TBW)
- Extracellular water (ECW)
- Intracellular water (ICW)
- Muscle mass
- Fat mass
- Protein
- Visceral fat area (VFA)
- Hepatic fat

**Laboratory Measures (12 features):**
- Glucose
- Total cholesterol, HDL, LDL, Triglycerides
- AST, ALT, ALP (liver enzymes)
- Creatinine, GFR (kidney function)
- CRP (inflammation marker)
- Hemoglobin
- Vitamin D

### Target Variable

**Gallstone** - Binary gallstone disease diagnosis
- 0 = No gallstone disease
- 1 = Gallstone disease

## Running the Sample

```bash
cd samples/gallstone
python main.py
```

## Measurements
	Sigmoid a	Sigmoid b	Train threshold		Weight penalty	Optimize threshold		Accuracy	AUPRC
	---------	---------	---------------		--------------	-------------------		--------	-----
	4		-1		0.9			1e-4		0.154				63.04%		0.7849
	1		0		0.9			1e-3		0.087				64.91%		0.7940
	1		0		0.5			1e-3		0.382				76.09%		0.8281
	4		-1		0.5			1e-3		0.320				73.29%		0.8421
5	4		-1		0.9			1e-3		0.625				78.26%		0.8703
	4		-1		0.8			1e-3		0.394				74.53%		0.8400
	4		-1		0.95			1e-3		0.334				69.57%		0.8409
7	4		-1		0.9			1e-3(88/4.0)	0.42				77.33%		0.8533
8	4		-1		0.9			1e-3(42/4.0)	0.382				78.26%		0.8424
9	4		-1		0.9			1e-3		0.309				73.60%		0.8322
	4		-1		0.9			1e-2		0.309				73.60%		0.8322