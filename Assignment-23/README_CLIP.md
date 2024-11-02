# Image Search using CLIP Model

Contrastive Language-Image Pre-training (CLIP) is a multimodal learning architecture developed by OpenAI. 
It learns visual concepts from natural language supervision.

CLIP is designed to predict which N × N potential (image, text) pairings within the batch are actual matches. 
To achieve this, CLIP establishes a multi-modal embedding space through the joint training of an image encoder 
and text encoder. The CLIP loss aims to maximize the cosine similarity between the image and text embeddings 
for the N genuine pairs in the batch while minimizing the cosine similarity for the N² − N incorrect pairings. 
The optimization process involves using a **symmetric cross-entropy loss function** that operates on these 
similarity scores. 

---

## What is contrast learning?

Contrastive learning is a method where we teach an AI model to recognize similarities and differences of a 
large number of data points.

Imagine you have a main item (the “anchor sample”), a similar item (“positive”), and a different item (“negative sample”). 
The goal is to make the model understand that the anchor and the positive item are alike, 
so it brings them closer together in its mind, while recognizing that the negative item is different and 
pushing it away.

---

## Architecture

ClIP uses two separate architectures as the backbone for encoding vision and text datasets:

- image_encoder: Represents the neural network architecture (e.g., ResNet or Vision Transformer) responsible for encoding images.
- text_encoder: Represents the neural network architecture (e.g., CBOW, BERT, or Text Transformer) responsible for encoding textual information.

For the image encoder the authors tried two different models, a ResNet-50 and a Vision Transformer (ViT). 
The largest ResNet model, RN50x64, took 18 days to train on 592 V100 GPUs while the largest Vision Transformer took 12 days on 256 V100 GPUs.

In Linear Algebra, one common way to measure if two vectors are of similar characteristics (they are like each other) is to calculate their dot product (multiplying the matching entries and take the sum of them); if the final number is big, they are alike and if it is small they are not (relatively speaking)!

Okay! What I just said is the most important thing to have in mind to understand this loss function. Let's continue. We talked about two vectors, but, what do we have here? We have image_embeddings, a matrix with shape (batch_size, 256) and text_embeddings with shape (batch_size, 256). Easy enough! it means we have two groups of vectors instead of two single vectors. How do we measure how similar two groups of vectors (two matrices) are to each other? Again, with dot product (@ operator in PyTorch does the dot product or matrix multiplication in this case). To be able to multiply these two matrices together, we transpose the second one. Okay, we get a matrix with shape (batch_size, batch_size) which we will call logits. (temperature is equal to 1.0 in our case, so, it does not make a difference. You can play with it and see what difference it makes. Also look at the paper to see why it is here!).

Let's consider what we hope that this model learns: we want it to learn "similar representations (vectors)" for a given image and the caption describing it. Meaning that either we give it an image or the text describing it, we want it to produce same 256 sized vectors for both.

**Training**
- Input: A bunch of image-caption(text) pairs (all encoded to be vectors).
- Output: The “cosine similarity” scores between all the image vector and caption vector combinations.
- Objective function: A contrastive function that will modify the weights of the model such that correct image-caption pairs get a high similarity score, and incorrect pairs get low similarity scores.

The training steps include:
- Embed the image with the image encoder and embed the text with the text encoder.
- The image and text embeddings will come from different models and will have different dimensions, so project them (by multiplying with a learnt projection matrix) into the same joint multimodal embedding space. For instance, np.dot(I_f, W_i) multiplies a matrix of size [n, d_i] with a matrix of size [d_i, d_e] which results in a projected matrix of size [n, d_e].
- Normalise the new embedding vectors. This turns them into unit vectors.
- Calculate the matrix of dot products.
- Calculate the cross entropy loss for each row and column and divide by 2, since each pair would be calculated twice.
  
**Inference**

- Inputs: the vector for a single image, and the vectors for a bunch of different possible text captions.
- Output: the similarity scores of the single image to all the different text captions.
- Goal: select the text which has the highest similarity with the image.

---

## Applications:

1. *Zero-Shot Image Classification*
One of the most impressive features of CLIP is its ability to perform zero-shot image classification. This means that CLIP can classify images it has never seen before, using only natural language descriptions.

2. *Multimodal Learning*
Another application of CLIP is its use as a component of multimodal learning systems. These can combine different types of data, such as text and images.
For instance, it can be paired with a generative model such as DALL-E. Here, it will create images from text inputs to produce realistic and diverse results. Conversely, it can edit existing images based on text commands, such as changing an object’s color, shape, or style. This enables users to create and manipulate images creatively without requiring artistic skills or tools.

3. *Image Captioning*
CLIP’s ability to understand the connection between images and text makes it suitable for computer vision tasks like image captioning

4. *Data Content Moderation*
Content moderation filters inappropriate or harmful content from online platforms, such as images containing violence, nudity, or hate speech. CLIP can assist in the content moderation process by detecting and flagging such content based on natural language criteria.

5. *Deciphering Blurred Images*
In scenarios with compromised image quality, such as in surveillance footage or medical imaging, CLIP can provide valuable insights by interpreting the available visual information in conjunction with relevant textual descriptions.

---

## References

1. https://viso.ai/deep-learning/clip-machine-learning/
2. https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2

---

