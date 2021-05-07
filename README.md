python train.py <br>
python eval.py  test.jpg<br>

change object list <br>
1. download git<br>
2. edit train.py eval.py <br>
  Add the object you want<br>
  ex)<br>
  NUM_CLASSES = 9 <br>
  class_idx_to_name = ['h', 'f', 'm', 'p', 'c', 'o']<br>
  <br>
3. mkdir  train_data checkpoints logs<br>
4. train_data file upload<br>
    filename.xml,  filename.jpg<br>
    Reference : vocedit  (https://github.com/jangsooyoung/vocedit)
<br>
5. train.py<br>
    generation checkpoints/~~~.h5<br>
6. eval.py input_image.jpg<br>

