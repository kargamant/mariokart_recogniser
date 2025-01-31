# mariokart_recogniser
Recogniser of Boo-Huu in mariokart using two approaches: Haar cascades and YOLO5.  
They were trained and compared through this project on different sets of images and even videos.  
My research and basic information about this project [report.pdf](https://github.com/kargamant/mariokart_recogniser/blob/dc84a38c95d1e57eaf998624df63b5f6cd86d317/report.pdf)  
### Image recognition and training  
Basic usage.  
```
$ python main.py -yolo --one_file boo_image.jpg
```
```
$ python main.py -haar --one_file boo_image.jpg
```
To make it recognise images on test set run one of the following two commands should be run.  
```
$ python main.py -yolo -test -res results_dir
```
```
$ python main.py -haar -test -res results_dir
```
> -res option can be omitted if you want results to be shown in separate window and not saved.
For more information run it with --help option.  
### Video stream recognition  
Basic usage.  
```
$ python video_stream.py --yolo --stream 2 -res processed_gp.mp4
```
```
$ python video_stream.py --haar --stream 2 -res processed_gp.mp4
```
> Also you can set stream to a file name of video you want to process instead of number of stream.
  
For more information run it with --help option.  

### Materials  
[**Dataset**](https://universe.roboflow.com/hope-demigod-g0ixy/mario-kart-8-deluxe-buu-huu-detection)  
[**Yolo_model**](https://github.com/ultralytics/yolov5)  
