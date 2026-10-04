"""Select the 1D-CNN readout on the VALIDATION set only."""
import os, sys, json; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np, tensorflow as tf
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import build, WINDOW
from models import cnn1d
d, m = build(); (Xtr,ytr),(Xva,yva) = d["train"], d["val"]
out={}
for ro in ["gap","last"]:
    v=[]
    for s in [0,1,2]:
        tf.keras.utils.set_random_seed(s)
        mod=cnn1d(WINDOW,Xtr.shape[2],ro); mod.compile(tf.keras.optimizers.Adam(1e-3),loss="mse")
        mod.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=100,batch_size=64,verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)])
        p=mod.predict(Xva,verbose=0).ravel()*m["y_sd"]+m["y_mu"]; y=yva*m["y_sd"]+m["y_mu"]
        v.append(float(np.mean(np.abs(p-y)))); print(ro,s,v[-1],flush=True)
    out[ro]=[float(np.mean(v)),float(np.std(v))]
json.dump(out,open("results/cnn_readout_val.json","w")); print(out)
