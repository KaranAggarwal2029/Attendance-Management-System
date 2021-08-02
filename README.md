# AttendanceManagementSystem
       
Face recognition is really a series of several related problems:
  1.	First, look at a picture and find all the faces in it
  2.	Second, focus on each face and be able to understand that even if a face is turned in a weird direction or in bad lighting, it is still the same person.
  3.	Third, be able to pick out unique features of the face that you can use to tell it apart from other people— like how big the eyes are, how long the face is, etc.
  4.	Finally, compare the unique features of that face to all the people you already know to determine the person’s name.


## Face Recognition — Step by Step
  **Step 1:** Finding all the Faces
    The first step in our pipeline is face detection. We need to locate the faces in a photograph before we can try to tell them apart!
    Face detection is a great feature for cameras. When the camera can automatically pick out faces, it can make sure that all the faces are in focus before it takes the picture. But we’ll use it for a different purpose — finding the areas of the image we want to pass on to the next step in our pipeline.
    We’re going to use a method called Histogram of Oriented Gradients — or just HOG for short.
    To find faces in an image, we’ll start by making our image black and white because we don’t need color data to find faces:

  **Step 2:** Posing and Projecting Faces
    Whew, we isolated the faces in our image. But now we have to deal with the problem that faces turned different directions look totally different to a computer:

    To account for this, we will try to warp each picture so that the eyes and lips are always in the sample place in the image. This will make it a lot easier for us to compare faces in the next steps.
    To do this, we are going to use an algorithm called face landmark estimation. The basic idea is we will come up with 68 specific points (called landmarks) that exist on every face — the top of the chin, the outside edge of each eye, the inner edge of each eyebrow, etc. Then we will train a machine learning algorithm to be able to find these 68 specific points on any face:

  **Step 3:** Encoding Faces
    Now we are to the meat of the problem — actually telling faces apart. 
    The simplest approach to face recognition is to directly compare the unknown face we found in Step 2 with all the pictures we have of people that have already been tagged. When we find a previously tagged face that looks very similar to our unknown face, it must be the same person. 
    What we need is a way to extract a few basic measurements from each face. Then we could measure our unknown face the same way and find the known face with the closest measurements. For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc. 

  **Step 4:** Finding the person’s name from the encoding
    This last step is actually the easiest step in the whole process. All we have to do is find the person in our database of known people who has the closest measurements to our test image.
    You can do that by using any basic machine learning classification algorithm. No fancy deep learning tricks are needed. We’ll use a simple linear SVM classifier, but lots of classification algorithms could work.
    All we need to do is train a classifier that can take in the measurements from a new test image and tells which known person is the closest match. Running this classifier takes milliseconds. The result of the classifier is the name of the person!.

