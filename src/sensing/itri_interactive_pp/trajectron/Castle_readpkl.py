import dill
with open('../experiments/processed/nuScenes_val_full.pkl', 'rb') as f:
        train_env = dill.load(f, encoding='latin1')
        print(train_env.scenes)
