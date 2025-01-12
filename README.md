# Cancer Diagnosis Using Machine Learning Classification Models

A comprehensive analysis of breast cancer diagnostic data using advanced machine learning classification techniques, demonstrating the application of computational methods in medical diagnosis.

## Project Overview

This project implements multiple machine learning classification models to analyze breast cancer diagnostic data, comparing their effectiveness in distinguishing between benign and malignant tumors based on cellular characteristics.

### Biological Context

Breast cancer diagnosis often begins with Fine Needle Aspiration (FNA), a minimally invasive procedure where cellular samples are extracted and analyzed microscopically. The cellular features examined include:

1. **Nuclear Characteristics**
   - Clump Thickness: Indicates cell layering
   - Uniformity (Cell Size/Shape): Cancer cells often show irregular sizes/shapes
   - Bare Nuclei: Presence of exposed nuclei (common in malignant cells)
   - Bland Chromatin: Describes nuclear texture patterns
   - Normal Nucleoli: Size and quantity of nucleoli

2. **Cellular Organization**
   - Marginal Adhesion: Cell-to-cell adherence
   - Single Epithelial Cell Size: Size consistency
   - Mitoses: Rate of cell division

## Dataset Details

The dataset (`Data.csv`) from the UCI Machine Learning Repository contains digitized breast mass FNA data:

- **Total Samples**: 699 cases
- **Features**: 10 numeric attributes (1-10 scale)
- **Classification**: Binary (2=benign, 4=malignant)
- **Missing Values**: None (cleaned dataset)

### Feature Descriptions

| Feature | Biological Significance | Scale Interpretation |
|---------|------------------------|---------------------|
| Clump Thickness | Indicates multi-layered cell growth | 1-10 (higher = more concerning) |
| Uniformity of Cell Size | Consistency of cell sizes | 1-10 (higher = more variable) |
| Uniformity of Cell Shape | Consistency of cell shapes | 1-10 (higher = more irregular) |
| Marginal Adhesion | Loss of cell cohesion | 1-10 (higher = less adhesion) |
| Single Epithelial Cell Size | Cell size relative to normal | 1-10 (higher = larger cells) |
| Bare Nuclei | Presence of naked nuclei | 1-10 (higher = more frequent) |
| Bland Chromatin | Nuclear texture | 1-10 (higher = more irregular) |
| Normal Nucleoli | Prominence of nucleoli | 1-10 (higher = more prominent) |
| Mitoses | Cell division frequency | 1-10 (higher = more frequent) |

## Implementation

Initially, multiple classification models were tested on the dataset:
- Logistic Regression (~91% accuracy)
- K-Nearest Neighbors (~92% accuracy)
- Naive Bayes (~89% accuracy)
- Random Forest (~93% accuracy)
- Support Vector Machines with various kernels (~90-93% accuracy)
- Decision Trees (~95-96% accuracy)

Based on performance metrics, the project retains the two highest-performing models:

### 1. Decision Tree (`decision_tree_classification.py`)
- Hierarchical binary decisions
- Highest accuracy at ~95-96%
- Highly interpretable results
- Natural handling of feature importance

### 2. Kernel SVM (`kernel_svm.py`)
- Uses Radial Basis Function (RBF) kernel
- Second-highest accuracy at ~93-94%
- Effective for non-linear classification
- Handles complex decision boundaries

## Results & Analysis

Our comparative analysis reveals:

1. **Model Performance Hierarchy**
   - Decision Tree Classification: 95-96% accuracy (best performer)
   - Kernel SVM: 93-94% accuracy (second-best)
   - Other tested models: <93% accuracy (excluded from final implementation)

2. **Key Findings**
   - Decision Trees consistently outperformed other models
   - Feature importance analysis reveals Uniformity of Cell Size and Shape as critical indicators
   - Low false-positive rate crucial for clinical application
   - Simpler models (like Logistic Regression) underperformed, suggesting non-linear relationships in the data

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CancerDiagnosisWithMLClassificationModels.git
```

2. Install requirements:
```bash
pip install sklearn pandas numpy matplotlib
```

3. Run the models:
```bash
python kernel_svm.py
python decision_tree_classification.py
```

## Future Improvements

- Integration of additional biomarkers
- Implementation of ensemble methods
- Cross-validation with external datasets
- ROC curve analysis for model comparison

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Dataset
2. Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis.
3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.