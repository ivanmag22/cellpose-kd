# Cellpose-KD
Useful guide to understand the code and what you need to consider for your scripts:

## Requirements
pip/conda install:
- numpy
- torch
- matplotlib
- tqdm
- numba
- torchvision
- tifffile
- sklearn
- skimage
- scipy
- pyqtgraph
- PyQt5
- natsort
- imagecodecs
- roifile
- fastremap

## Command
- **Cellpose-KD**
```bash
python cellpose/test_cellpose.py --input "$file" --diameter 25 --dilation --upsample_compr --nbase 16 32 64 128 --depthwise --model_path models/40x/cp-kd/Best_model_epoch_20_lr1.00e-03_d25.0.pth --output "$output_dir"
```
- **FastCellpose-KD**
```bash
python cellpose/test_cellpose.py --input "$file" --diameter 25 --fastcp --nbase 16 32 64 128 --model_path models/40x/fastcp-kd/Best_model_epoch_55_lr1.00e-03_d25.0.pth --output "$output_dir"
```
- **DAccuracy**
```bash
python daccuracy/package/daccuracy/cli/main.py --gt "gt.csv" --rGcF --dn "dn.tif" -s
```

## Teacher-Student learning
To deal with unlabelled dataset, the intuition is to give as input an image to Cellpose with model_type="nuclei" (or a variant obtained as mentioned in the previous chapter) and to a smaller variant and to let the second one learn from the first model.
I used the standard loss function from original Cellpose. But, I changed the value of weighted lambda parameter in gradient flows contribution. I set this parameter to 1 with the following motivation: during training we are working with outputs from two models, so we need to compute directly the difference between teacher and student's outputs and not the same. So, the resulting method is the following:
```python
def _loss_fn_seg1(lbl, y, device):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 1.0 * torch.from_numpy(lbl[:, 1:]).to(
        device
    )  # 5.0 * torch.from_numpy(lbl[:, 1:]).to(device)
    loss = criterion(y[:, :2], veci)
    loss /= 2.0
    loss2 = criterion2(y[:, -1], torch.from_numpy(lbl[:, 0] > 0.5).to(device).float())
    loss = loss + loss2
    return loss
```
The [issue](https://github.com/MouseLand/cellpose/issues/1120) solved (partially) my doubts: in our case we need to set this value to 1 since we are dealing with outputs from two models and we are not comparing a GT mask (whose flows are obtained with *dynamics.py / masks_to_flows()* method and its values are in [-1,1] range) and model output.

## Models
Useful guide to models, since there are too many. We will see only models/cellpose folder.
- **Cellpose-KD**: our final version
    - 40X dataset:
        - models/40x/cp-kd/Best_model_epoch_20_lr1.00e-03_d25.0.pth
    - 20X dataset:
        - models/20x/cp-kd/Best_model_epoch_210_lr1.00e-03_d25.0.pth
- **FastCellpose-KD**
    - 40X dataset:
        - models/40x/fastcp-kd/Best_model_epoch_55_lr1.00e-03_d25.0.pth
    - 20X dataset:
        - models/20x/fastcp-kd/Best_model_epoch_240_lr1.00e-03_d25.0.pth


## GUI
To run the GUI you need to type the following in the command line:
```
python cellpose/cellpose/__main__.py
```
Then open *Models > Training instructions* to understand the step that you need to train your model with human-in-the-loop approach.
Otherwise for more simple operations, like obtaining the instance segmentation mask, you can select the model and then run it. But maybe for model_type="nuclei" setting a value for diameter is not possible.

## Code
- **cellpose**: generalist algorithm for semantic and instance segmentation of cells.
    - *cellpose/cellpose/models.py*: it contains classes and methods useful for the definition of the model. We can define it as a two-stage model: the first part is represented by the network and it computes the output in terms of cellprob, gradient flows and style vector; the second part computes the correspondant mask from the previous output. It is possible to execute only the first part by setting *compute_masks=False*.
    - *cellpose/cellpose/resnet_torch.py*: this script contains the original classes for the U-Net like architecture. Given nbase=[32,64,128,256] which represents the number of channels during the downsample part (the reverse for upsample part), each encoding/decoding unit contains four convolutional layers and a pooling layer (downsample) or unpooling layer (upsample). I worked mostly on *resdown* and *resup* classes, but later I had touched also *downsample* and *upsample* classes.
    - *cellpose/cellpose/dilated_resnet.py*: network compression applied only on encoding path. It contains in downsample a lower number of convolution layers (from 4 to 2 per each encoding unit) and in order to get a good result I introduced dilation in order to have a feature map with a higher receptive field, so that can summarize well the input and the previous feature maps (but previously activation function was also applied!). Dilation is applied only on the second convolution layer of the unit.
    - *cellpose/cellpose/upsample_resnet.py*: network compression applied only on decoding path. This version contains not only the upsampling layer, but also transposed convolutional layers.  I alternate both layers.
    - *cellpose/cellpose/downup_resnet.py*: given the compression made in downsample (dilated_resnet.py) and in upsample part (upsample_resnet.py), the number of parameters and MACs decreases in a good way. 
    - *cellpose/cellpose/resnet_torchv2.py*: it is a version of resnet_torch.py where I replace traditional convolutional layers with depthwise separable convolutional ones, so convolution operation is decomposed into simpler parts, in order to decrease the amount of parameters and of MACs. At the same time a good training is needed in order to find a good set of weights for this simple model.
    - *cellpose/cellpose/downup_resnetv2.py*: this script contains the architecture compressed in both encoding and decoding paths; moreover, it contains both depthwise separable convolution layer and transposed version.
    - *cellpose/cellpose/fastcp_resnet_torch.py*: it contains FastCP network definition. Here, you can see that by enabling residual_on you will adopt Cellpose architecture with two layers per encoding/decoding unit, while by setting it to False you will adopt transposed convolutional layers instead of upsampling layers. Furthermore, style branches can be "cut" by disabling style_on. I do not know too much about concatenation.
    - *cellpose/cellpose_metrics.py*: script useful to get number of parameters and number of MACs (Multiply-Accumulate Computations) given a tensor of shape (1,2,1024,1024) (it is possible to change number of slices, width and height by passing different values to z, x, y arguments respectively). You can tell if you want to get these statistics for Cellpose, Cellpose with dilation (downsample compression), Cellpose with transposed convolutional layers (upsample compression), Compressed Cellpose, Cellpose with depthwise separable convolutions, pruned Cellpose etc.
    - *cellpose/test_cellpose.py*: given in input a 2D or 3D sample, it computes the prediction and also the mask and it returns also the inference time. The mask will be saved in the directory where the script is ran or with --option argument you can specify the folder where to save the mask. To not generate the mask, you must enable --no_mask argument. You can use several variants of Cellpose and you can pass the desired value for diameter. 
    - *cellpose/train_kd_batch.py*: script that contains useful methods for teacher-student learning (with teacher's weights fixed). As *teacher**, usually we use generalist Cellpose with model_type="nuclei" as teacher, but if you want you can load a pretrained model (always compatible with "nuclei" model type). As **student** you can use the model you prefer, by specifying the argument for Dilated Cellpose, Upsample Compressed Cellpose, Compressed Cellpose, Cellpose with depthwise separable convolutions and other variants. Furthermore, you can modify the architecture by defining the number of feature maps per layer with *nbase*. You can always load weights from pre-trained model by defining its path. You can use other values for batch size, learning rate,, number of epochs optimizer etc.; you can adopt the loss function that you want (standard one or the one with lambda=1 that seems to work better than the previous one). If your training was interrupted, you can resume by specifying the number of epoch, in order to pick that model and to continue the training. You can decide what is the frequency of validation during training. Model are saved every checkpoint_step epoch(s) (by default every 1 epoch). An important hyper-parameter is the diameter, that represents the average size of objects to detect; in this case both teacher and student have the same value in order to have a coherent comparison. The comparison is made in terms of IoU (binary classification, background or nuclei pixels) and it considers only cellprob because they are torch tensors and on GPU it is more difficult to handle masks that are numpy.ndarrays (if you want to see the metrics, go to cellpose/cellpose/metrics.py). If you want to train only a set of layers you can decide to freeze all layers with the exception of transposed convolution ones, to freeze layers from downsample part or to freeze layers from upsample part; freezing some layers can be useful to train only the layers that you want to optimize and not the whole network. You can load only the weights from downsample part, weights from upsample part or both. Moreover you can decide to keep or remove style branches (they could be useless). If you dataset is composed by 3D samples do not pass anything, otherwise pass --slices if you want to train on slices or --patches if you want to train on patches (this script works really well with patches; it can work also with slices, but you can encounter Out-Of-Memory error; it does not contain methods for training on 3D samples); you need to specify which is the folder to get the samples and how many samples take and if you want if to take an equal number for each phenotypic class. It is possible to adopt data augmentation to increase virtually your dataset.
    - *cellpose/train_kd_batch_fastcp.py*: it is like cellpose/train_kd_batch.py, but it adopts as student FastCellpose model (and as teacher Cellpose model). You can set to True some flags for resnet layers, style branches and concatenation operations (we can see these later).
    - *cellpose/utils.py*: it contains polinomial learning rate scheduler method.
- **utils**:
    - *label_correction.py*: this script allows to obtain an image (2D/3D) where each label is not repeated. The output will have the suffix "_lab_corrected". It is useful because sometimes Cellpose segmentation maps are not allowed as detection for DAccuracy due to the repetition of some labels (one constraint of DAccuracy is that each detected object must have an unique value as label)