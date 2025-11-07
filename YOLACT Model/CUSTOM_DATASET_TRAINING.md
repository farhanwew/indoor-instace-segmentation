# Training YOLACT on a Custom Dataset

This guide provides step-by-step instructions on how to train YOLACT on your own custom dataset for instance segmentation.

## Step 1: Data Preparation (COCO Format)

YOLACT expects your dataset to be in the COCO format. This involves organizing your images and creating a JSON annotation file.

### Directory Structure

Organize your dataset as follows:

```
/path/to/your/dataset/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── annotations.json
```

### `annotations.json`

This JSON file contains the annotations for your images. It has three main sections:

*   `"images"`: A list of your images with their file names, height, and width.
*   `"categories"`: A list of the object classes you want to detect.
*   `"annotations"`: A list of every object instance in your dataset, including bounding boxes and segmentation masks.

**Example `annotations.json`:**

```json
{
  "images": [
    {
      "id": 1,
      "width": 640,
      "height": 480,
      "file_name": "000001.jpg"
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "cat",
      "supercategory": "animal"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 250],
      "segmentation": [
        [100, 150, 300, 150, 300, 400, 100, 400]
      ],
      "area": 50000,
      "iscrowd": 0
    }
  ]
}
```

**Note:** Creating this file manually can be tedious. Tools like [LabelMe](http://labelme.csail.mit.edu/Release3.0/), [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/), or [COCO Annotator](https://github.com/jsbroks/coco-annotator) can help you create these annotations.

## Step 2: Modify the Configuration (`yolact/data/config.py`)

You need to tell YOLACT about your new dataset by modifying the `yolact/data/config.py` file.

### a. Add a New Dataset Configuration

Add the following code to `yolact/data/config.py` to define your custom dataset. Make sure to replace the placeholder paths with the actual paths to your dataset.

```python
my_custom_dataset = dataset_base.copy({
    'name': 'My Custom Dataset',

    'train_images': '/path/to/your/dataset/images/',
    'train_info': '/path/to/your/dataset/annotations.json',

    'valid_images': '/path/to/your/dataset/images/',
    'valid_info': '/path/to/your/dataset/annotations.json',

    'class_names': ('my_class_1', 'my_class_2', 'my_class_3'),
})
```

### b. Add a New Model Configuration

Next, add a new model configuration that uses your custom dataset. Add the following code to `yolact/data/config.py`:

```python
yolact_custom_config = yolact_base_config.copy({
    'name': 'yolact_custom',

    # Dataset stuff
    'dataset': my_custom_dataset,
    'num_classes': len(my_custom_dataset.class_names) + 1,

    # You can overwrite other parameters here, like learning rate, number of iterations, etc.
    # 'lr': 1e-4,
    # 'max_iter': 100000,
})
```

## Using Pre-trained Weights

You can specify a weight file to start training from, which is useful for resuming an interrupted training session or for fine-tuning a model.

### Resuming Training

Use the `--resume` argument to load a full checkpoint and continue training:

```bash
python train.py --config=yolact_custom_config --resume=weights/yolact_custom_xxxx.pth
```

You can also use `--resume=latest` to automatically resume from the most recent checkpoint for your configuration.

### Using Pre-trained Backbone Weights

YOLACT automatically loads pre-trained weights for the backbone network (e.g., ResNet, VGG) when you start a new training run. The path to these weights is defined in the `path` field of the backbone configuration in `data/config.py`. Make sure the specified weight file exists in the `weights/` directory.

## Training with YOLACT++

The training procedure for YOLACT++ is almost the same, but you need to use a YOLACT++ configuration and compile the DCNv2 extension first.

### a. Compile DCNv2

Before you can train or run a YOLACT++ model, you must compile the DCNv2 extension. From the `yolact` directory, run:

```bash
cd external/DCNv2
python setup.py build develop
```

### b. Create a YOLACT++ Custom Configuration

In `yolact/data/config.py`, create a new configuration that inherits from a YOLACT++ base configuration (e.g., `yolact_plus_base_config`):

```python
yolact_plus_custom_config = yolact_plus_base_config.copy({
    'name': 'yolact_plus_custom',

    # Dataset stuff
    'dataset': my_custom_dataset,
    'num_classes': len(my_custom_dataset.class_names) + 1,

    # You can overwrite other YOLACT++ specific parameters here if you want
})
```

## Step 3: Train the Model

Now you are ready to train your model. Run the `train.py` script with your new configuration:

```bash
# For standard YOLACT
python train.py --config=yolact_custom_config

# For YOLACT++
python train.py --config=yolact_plus_custom_config
```

The script will periodically save model weights to the `yolact/weights/` directory.

## Step 4: Validation and Evaluation

To evaluate your model's performance on your validation set, use the `eval.py` script. Remember to use the correct configuration for your model.

```bash
python eval.py --trained_model=weights/yolact_custom_xxxx.pth --config=yolact_custom_config --score_threshold=0.15 --top_k=15
```

*   `--trained_model`: Path to the saved weight file you want to evaluate.
*   `--config`: The name of your custom model configuration.
*   `--score_threshold` and `--top_k`: Optional parameters to filter detections.

This will output the mean Average Precision (mAP) for both bounding boxes and masks.

## Step 5: Testing and Inference

To run inference on new images and visualize the segmentation results, you can also use `eval.py`.

### Display Results on an Image

```bash
python eval.py --trained_model=weights/yolact_custom_xxxx.pth --config=yolact_custom_config --score_threshold=0.15 --top_k=15 --display /path/to/your/image.jpg
```

### Save the Output Image

```bash
python eval.py --trained_model=weights/yolact_custom_xxxx.pth --config=yolact_custom_config --score_threshold=0.15 --top_k=15 --output_image /path/to/output/image.jpg /path/to/your/image.jpg
```
