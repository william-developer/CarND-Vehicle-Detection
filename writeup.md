
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image4]: ./output_images/tests.png
[image5]: ./output_images/heatmap.png


---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters.
Firstly, I use parameters like that
```python
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" 
color_space = "YUV"
```
and the accuracy of SVC was 0.9964.
Then I tried five color spaces
```python
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel="ALL" 
color_spaces = ["RGB","HSV","LUV","HLS","YUV","YCrCb"]


for color_space in color_spaces:
    print("="*50)
    print("Color space : ", color_space)
    SVC_HOG(color_space=color_space, orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, hog_channel=hog_channel,debug=False)
```
The result was that YUV,I got the best Accuracy could improve accuracy.The accuracy was 0.9964.
After that,I tried orient with [8,9,10,11].The best orient was 9 and 11.
Finally when orient was 11 and color space was YUV,I got the best accuracy which was 0.9964.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features and color features.
Firstly I extract spatial features, hist features and hog features.Then I conbined them.
After that I nomalized it.To get better result,I used GridSearchCV to tune LinearSVC parameters.
```python
X_train = np.vstack((car_features[car_train_indices],notcar_features[notcar_train_indices])).astype(np.float64)   
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = np.vstack((car_features[car_test_indices],notcar_features[notcar_test_indices])).astype(np.float64) 
X_test = standard_scaler.transform(X_test) # use same fit as training data

# labels vector
y_train = np.hstack((np.ones(len(car_train_indices)), np.zeros(len(notcar_train_indices))))
y_test = np.hstack((np.ones(len(car_test_indices)), np.zeros(len(notcar_test_indices))))

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
#svc = LinearSVC()
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(LinearSVC(), param_grid={'C':np.logspace(-3,-2,5)})
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this.
As part of the picture was useless,I cut the effective area.
```python
# sliding window scales and the search y-range
y_range = {1.0 : (380,508), 
           1.5 : (380,572), 
           2.0 : (380,636), 
           2.5 : (380,700)}
```
Firstly,I calculated the steps to slid window search.Then extracted HOG for this patch and the image patch.
Then I normallize the combination of spatial features, hist features and hog features.
After that I got a prediction. 
Finally,I draw boxes if there was a vehicle.
```python
for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins,bins_range=bins_range)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features,
                                                          hog_features)).reshape(1, -1)) 
            test_prediction = clf.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw,ytop_draw+win_draw+ystart),
                              (0,0,1),6)
                bbox_list.append(((xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
I tried some color spaces and histograms parameters and tune the linear sve parameter using GridSearchCV.
Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_videos_output/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


### Here are six frames and their corresponding heatmaps and the resulting bounding boxes  :

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The first challenge was the training set and the test set.The GTI* folders contained time-series data and even a randomized train-test split would be subject to overfitting.I extracted the time-series tracks from the GTI data and separating the images manuallymanually to make sure train and test images set were sufficiently different from one another.

The second challenge was the method `sklearn.preprocessing.StandardScaler()`.I got some errors that about different dimensions.

The parameters I chose could make the model work well.But there was some errors in few images.On the one hand, it was defect of the model which could not get one hundred percent prediction.On the other hand,it was the complexity of the road system,such as light and terrain.

If the continuity of the video frame was taken into account, the model would be more robust.



```python

```
