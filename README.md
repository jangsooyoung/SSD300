python train.py 
python eval.py  test.jpg

change object list 


1. download git
2. edit train.py eval.py 
  NUM_CLASSES = 9
  class_idx_to_name = ['h', 'f', 'm', 'p', 'c', 'o']
  
3. mkdir  train_data checkpoints logs
4. train_data file upload
    filename.xml,  filename.jpg
    Reference : vocedit  (https://github.com/jangsooyoung/vocedit)

5. train.py
    generation checkpoints/~~~.h5
6. eval.py input_image.jpg

