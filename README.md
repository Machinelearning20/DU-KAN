
conda create -n seg python=3.10
conda activate seg 
pip install -r requirements.txt
```

**Tips A**:  pytorch=1.13.0, and the CUDA compile version=11.6. .

The resulted file structure is as follows.
```

├── inputs
│   ├── busi
│     ├── images
│           ├── xxxx.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── xxxx.png
|           ├── ...
│   ├── kvasir
│     ├── images
│           ├── xxxx.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── xxxx.png
|           ├── ...
│   ├── cvc
│     ├── images
│           ├── xxxx.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── xxxx.png
|           ├── ...
```



## ⏳Training Segmentation 

You can simply train it on a single GPU by specifing the dataset name ```--dataset``` and input size ```--input_size```.
```bash
python train.py --arch DUKAN --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_DUKAN  --data_dir [YOUR_DATA_DIR]
```

```bash
cd DU-KAN
python train.py --arch DUKAN --dataset busi --input_w 256 --input_h 256 --name busi_DUKAN  --data_dir ./inputs
```

