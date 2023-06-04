## Clothing Segmentation and Embedding

### Whats included here:
  * Some test images 
  * ```general_json2yolo.py``` utility file for converting fashionpedia dataset to YOLO style dataset 
  * ```utils.py```  random utility functions for dataset conversion
  * ```endpoint_utils.py``` utility functions for interacting with the endpoint once it is deployed 
  * ```yoloV8.ipynb``` YoloV8 training code and model useage code. Also image and mask preview code (draw a mask on image)
  * The ```cebrium``` directory contains everything you need to deploy the model. All you need to do is put in your private key and then run the first cell in ```CerebriumIntegration.ipynb```
  * ``main.py`` contains the necessary inference code for the model endpoint. Note it can download an image from the internet or use a base64 encdoded image

#### How to launch the model:
* get your private api key from [here](https://dashboard.cerebrium.ai/projects/p-5e206a4c/api-keys) and put it in the first cell in ```CerebriumIntegration.ipynb```
```
private_key = 'your key goes here'
```
* Run the cell below that - on my Mac I do that via ```shift+enter```. It'll take ~8 mins to deploy the first time
* The rest of that notebook shows you how to use that endpoint. It includes code for:
  * sending a url request (download image from url)
  * sending a base64 encoded image request (send a local image)
  * sending an embedding only request (for your existing database)
  * finding similar images using some CLIP utils 
  * encoding/decoding base64 strings because its finicky 

#### How did I train the segmentation model 
* Found a dataset called [fashionpedia](https://fashionpedia.github.io/home/Fashionpedia_download.html) containing ~40K people in clothing and segmentation masks
* The dataset had ALOT of irrelevant features, so I filtered the dataset down to only 27 classes (can see them in the .yaml file)
* I converted that dataset to a YOLO compatible dataset. You can check out more about [yolo here](https://docs.ultralytics.com/)
* I trained a YOLO model (```YOLOv8s-seg```) 120 epochs. I used AWS sagemaker to train it on a ```ml.g4dn.
  12xlarge```. It took ~36 hrs 
* I saved the weights and they're now in the ```cerebrium``` directory (```yoloWeights.pt```). They're loaded when the model is deployed 

#### How does the embedding work
* I use a pretrained CLIP model to do image embedding. I demo how you can use it in the ```CerebriumIntegration.ipynb``` file

#### Notes on the endpoint
* Average latency seems to be 250-300 ms. If you don't mind slower inference you could train a larger YOLO model. I did 
  not have the time or $$$ to do this 
* The endpoint can take in either a URL and download an image OR a base64 encoded image
* It returns 4 arrays depending on the input parameters. 
  * If you don't specify ```embedding_only``` it will run the image through the yolo model and then embed each individual mask
  * If you specify ```embedding_only``` it will only return a 1d array of the given image embedings (this is for your existing database)
  * I specify some opencv and necessary CLIP dependencies in the ```pkglist.txt```
  * Everything else in the ```requirements.txt``` file is pretty standard


#### Some Ideas for future improvement?
* Set it up to support batch inference i.e. pass in more than 1 image. I would guess ~50% of the client side latency is from network hops. Maybe you can do 5 images at a time or something
* Try a larger YOLO model for better accuracy or a smaller model for faster inference times. If you are really 
  interested in speed you could also serve the model at half precision  
* Experiment with different CLIP versions.  
* Get rid of numpy and PIL usage in the inference file and only use numpy to handle the images
* Return bounding boxes in addition to segmentation masks  
* Use something like Roboflow to create a dataset for your specific database. This will give you better results 
* Use a transformer model pre-trained on COCO i.e. [Beit](https://huggingface.co/docs/transformers/v4.29.
  1/en/model_doc/beit#transformers.BeitForImageClassification). This might give better accuracy than YOLO (hard to 
  say and would definitely be slower)
