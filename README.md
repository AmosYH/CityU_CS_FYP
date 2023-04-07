# Low Light Image Enhancement and Restoration for Face Detection Using Fusion of Multiple Algorithms

## Low Light Enhancement and Face Detection Flask Web Application

### Installation
* Install Nvidia Driver
* Install Docker 
* Configure Docker with Nvidia Driver
* Unzip the file weights.7z under Flask_Web_Application/weights directory
* Build Container Image by Executing Flask_Web_Application/build.sh

    ```sudo ./build.sh```

* Launch the application by Executing Flask_Web_Application/run.sh

    ```sudo ./run.sh```

* Access the application at [http://localhost:8080/](http://localhost:8080/)

### How to Stop the Web Application
* Execute Flask_Web_Application/stop.sh

    ```sudo ./stop.sh```

## SCI+Zero-DCE++
### Start

    cd SCI+Zero-DCE++
    python inference.py
The output images can be found under **SCI+Zero-DCE++/results**.
## HINet
### Installation

    cd HINet
    pip install -r requirements.txt
    python setup.py develop --no_cuda_ext
Please download the pretrained model of HINet-SIDD-1x following the instruction at **experiments/pretrained_models/README.md** and place the model on the same directory.
### Start
 
Execute the following in the terminal:

    python basicsr/demo.py -opt options/demo/demo.yml

The input images are put under **HINet/demo/DCE+++SCI**, and the output images are placed under **HINet/demo/Denoised**.

## RetinaFace
### Installation
Unzip the file weights.7z under **RetinaFace_FaceDetection/weights** directory.
### Start

    cd RetinaFace_FaceDetection
    python process.py
The input photos are in **RetinaFace_FaceDetection/data**, while output photos are under the folder **RetinaFace_FaceDetection/results**.

## Evaluation Metrics
Four Evaluation Metrics are implemented in Python to assess the performance of the outcomes. They are:
 - PSNR
 - SSIM
 - UQI
 - ERGAS
### How to Evaluate
**Evaluation_Metrics/gt** places the ground truth images. Images for evaluation are to be put into corresponding folders under **Evaluation_Metrics/pred**.

    cd Evaluation_Metrics
    python psnr_cal.py
    python ssim_cal.py
    python uqi_cal.py
    python ergas_cal.py

rescale.py is for rescaling output images because of possible inconsistent image sizes (due to the structure of the convolutional layers) between ground truth images and model result images.
