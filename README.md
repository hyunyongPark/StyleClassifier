# Kfashion Style classifier - Multi Label classification

### Model Description 
<table>
    <thead>
        <tr>
            <td>EfficientNet Model Architecture</td>
            <td>Activation Map</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/StyleClassifier/blob/main/img/architecture.png"/></td>
            <td><img src="https://github.com/hyunyongPark/StyleClassifier/blob/main/img/act_map.png"/></td>
        </tr>
    </tbody>
</table>



### Requirements
- python V  # python version : 3.8.13
- dgl==0.9.1
- tqdm
- torch==1.9.1
- torchvision==0.10.1
- torchaudio==0.9.1
- torchtext==0.10.1
- dask
- partd
- pandas
- fsspec==0.3.3
- scipy
- sklearn



### cmd running

The install cmd is:
```
conda create -n your_prjname python=3.8
conda activate your_prjname
cd {Repo Directory}
pip install -r requirements.txt
```
- your_prjname : Name of the virtual environment to create


If you want to proceed with the new training, adjust the parameters and set the directory and proceed with the command below.

The Training cmd is:
```

python3 style_train.py 

```

The testing cmd is: 
```

python3 Inference_inspection_kfashiontest.py 

```

The inference cmd is: 
```

python3 Inference.py 

```

### Test Result

###### Testset Distribution
<table>
    <thead>
        <tr>
            <td>testset fashion category</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/StyleClassifier/blob/main/img/distribution.png"/></td>
        </tr>
    </tbody>
</table>


- Model Performance Table

###### test performance
|Model|Class Num|Testset Num|Top3 Recall|
|---|---|---|---|
|Global Convolutional Network|10|41,178|*91.1%*|
|EfficientNet|23|118,483|**95.5%**|

|Class|Number|Top3 Recall|
|---|---|---|
|preppy|1,218|*89.8%*|
|resort|29,757|*94.0%*|
|punk|389|*91.9%*|
|classic|14,809|*91.4%*|
|military|1,656|*95.4%*|
|sporty|6,710|*94.4%*|
|retro|3,512|*93.5%*|
|oriental|1,704|*90.6%*|
|country|13,792|*93.1%*|
|hiphop|1,254|*88.0%*|
|hippy|2,644|*93.8%*|
|avantgarde|1,576|*89.8%*|
|modern|31,242|*93.6%*|
|romantic|28,976|*95.0%*|
|manish|3,382|*88.2%*|
|genderless|6,003|*92.8%*|
|kitsch|2,858|*90.6%*|
|tomboy|3,921|*88.7%*|
|street|134,410|*97.5%*|
|feminine|34,652|*93.9%*|
|western|665|*88.7%*|
|sophisticated|11,960|*91.9%*|
|sexy|3,714|*90.8%*|


- Example 
<table>
    <thead>
        <tr>
            <td>testset fashion category</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/StyleClassifier/blob/main/img/distribution.png"/></td>
        </tr>
    </tbody>
</table>
