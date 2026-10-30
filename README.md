# Air Quality Prediction Using Deep Learning: Next-Hour CO(GT) Forecasting
ICT 4442 Deep Learning Mini Project.

    pip install tensorflow pandas scikit-learn matplotlib

Phase 2 (interim, preliminary results):

    python src/train.py               # all 4 models x 3 seeds -> results/results.json
    python src/cnn_readout_ablation.py  # 1D-CNN readout selection (validation only)
    python src/make_figures.py        # figures/fig1-fig6

Phase 3 (final):

    python src/tune.py                # 6 configs x 2 seeds per model, ranked on VALIDATION MAE -> results/tuning/
    python src/final_eval.py          # selected config x 5 seeds, scored on test -> results/final/
    python src/error_analysis.py      # comparison table, error analysis, DM tests -> results/final_summary.json, figures/fig7-fig14

`tune.py` and `final_eval.py` take model names as arguments (e.g. `python src/tune.py LSTM`)
so the four models can run in parallel; set `TF_THREADS=3` to share the CPU.
All Phase 3 runs share one protocol (`src/protocol.py`): windows aligned to 48 h so
every window length is scored on the same target hours, same chronological split,
Adam, MSE loss, batch 64, early stopping on validation loss (patience 10).

Owners: MLP - Ashwin Singh Kushwaha | 1D-CNN - Mrittika Sarkar |
LSTM - Sagnik Chattopadhyay | Transformer - Tanish Sam Mathai.
Data: UCI Air Quality dataset (CC BY 4.0), doi:10.24432/C59K5F.
