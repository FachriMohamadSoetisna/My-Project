# My-Project
My past projects in data science practices.

Alphabet & picture keypoint detector
Basically determine which keypoint is from alphabet or which is from picture.
We use the openCV for generate the keypoints and descriptor from the dataset, and feed them
to the kMeans clustering algorithm to determine what type the keypoints are.
The dataset we used are a bunch of pictures of book page we took with a phone camera.
One of the usage of our project is for extracting information such as name, address, etc. from
ID card photo for user's validation.

Plagiarism checker
We build a simple plagiarism checker between documents such as academic writing, thesis, article, etc.
We used pairing system so that we can cover all possible document pairing in exchange of cost of time.
We used cosine distance to calculate the similarity between documents because the vectors that generated from text
data are usually sparse, and cosine distance support sparse data more than euclidean distance.
The main usage of this project is to check the similarity between documents so that if some documents had an unusual
similarity, the respective documents will be checked manually.
