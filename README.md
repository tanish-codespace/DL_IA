# Air Quality Prediction Using Deep Learning: Next-Hour CO(GT) Forecasting
ICT 4442 Deep Learning Mini Project.

    pip install tensorflow pandas scikit-learn matplotlib
    python src/train.py               # all 4 models x 3 seeds -> results/results.json
    python src/cnn_readout_ablation.py  # 1D-CNN readout selection (validation only)
    python src/make_figures.py        # figures/

Owners: MLP - Ashwin Singh Kushwaha | 1D-CNN - Mrittika Sarkar |
LSTM - Sagnik Chattopadhyay | Transformer - Tanish Sam Mathai.
Data: UCI Air Quality dataset (CC BY 4.0), doi:10.24432/C59K5F.
