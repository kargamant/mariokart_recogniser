# mariokart_recogniser
Recogniser of Boo-Huu in mariokart using two approaches: Haar cascades and YOLO5.  
My research and basic information about this project [report.pdf](https://github.com/kargamant/mariokart_recogniser/blob/dc84a38c95d1e57eaf998624df63b5f6cd86d317/report.pdf)  
### Image recognition  
Basic usage.  
```
$ python main.py -yolo --one_file boo_image.jpg
```
Or  
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

