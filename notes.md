- SGD, Stochastic Gradient Descent: Simply the process of updating the parameters
(weights) of a neural net. Specifically using the math formula of ...
    - Book, lesson 1
- Loss
    - Measuring model performance. Measuring how many things the model gets correct
    after training. Optimized for SGD to use to update weights
    - Book, lesson 1
    - Number that is higher when the model is incorrect, even higher when the model is
    more confident of its incorrect label. Also high when it has a correct label but
    less confident of the correct label.
    - Book, lesson 2
- Metric: Measuring model performance. Function to measure quality of model's
predictions, printed after each epoch. Optimized for human consumption to have
visibility into the training of the model and having a feel for how well it's doing.
    - Error Rate: percentage of values being predicted/classified correctly
    - Accuracy: 1.0 - error_rate
    - Book, lesson 1
- Loss vs Metric
    - Close, but different. Can be the same function used for both, or not.
    - Loss is for SGD to use to update weights, machine oriented
    - Metric is for humans to understand how model is performing
    - Book, lesson 1
- Dependent variable: Also known as the targets, labels, correct labels, y (vertical)
    - Independent variable is x (horizontal) (Book, lesson 2)
    - Book, lesson 1
- Fit: Synonym for train.
    - Book, lesson 1
- Models
    - Classification model: Predict a category, aka discrete values
    - Regression model: Predict numeric quantites like temperature or location.
        - Not linear regression model, that has a specific different meaning, bad
        practice to call a linear regeression model a 'regression model'
    - Book, lesson 1
- Three, not two, but three data sets: training set, validation set (aka development
set), test set. Make sure the test set the human doesn't look at either so the human
coming up with hyperparameters, doesn't inadvertently overfit the data accidentally.
- Overfitting hints
    - Validation accuracy is getting worse during training
    - Important but don't pre-optimize to avoid it, you might be ok
    - Book, lesson 1
- More layers in a neural net (resnet 18, 34, 50, 101, 152) will only be more accurate
if you have a correspondingly larger dataset, otherwise it is prone to overfit faster
and be worse than using fewer layers. With large amounts of data they can be more
accurate.
    - Book, lesson 1
- Model head: When using pre-trained models (you always should), the last layer is
specific to the original thing pre-trained for (like ImageNet dataset classification).
This layer gets removed, by fast.ai, and replaced by one or more new layers with
randomized weights, appropriate for the size of the dataset you're using. This is the
head.
    - Book, lesson 1
- Transfer learning: Using a pre-trained model for something other than what it was
originally trained for. i.e. using a pre-trained model for anthing other than what it
was origially trained on
    - Book, lesson 1
- Epoch: 
    - One complete pass through the training dataset.
    - How many times to look at each piece of data. How many times to look at each
    image. The number of layers in a neural net.
    - After each epoch it prints:
        - The epoch number, the training and validation set losses (the "measure of
        performance" used for training the model), and any metrics you've requested
        (error rate for images usually, or other).
    - Book, lesson 1
- Parameters: aka weights. Can be weights or .... The values the model updates to get
better accuracy.
    - Book, lesson 1
- fit() (fit_one_cycle()) vs fine_tune(), for fast.ai
    - fit() (fit_one_cycle()): all parameters (weights) are randomized, essentially starting from scratch
    - fine_tune(): use existing parameters (weights) that are on a pre-trained model
    already and just update them with a few more epochs of training on the dataset we're
    using it for now
        - fine_tune() does:
            1. Use one epoch to fit just those parts of the model necessary to get the
            new random head to work correctly with your dataset.
            2. Use the number of epochs requested when calling the method to fit the
            entire model, updating the weights of the later layers (especially the head)
            faster than the earlier layers (which, as we'll see, generally don't require
            many changes from the pretrained weights).
    - Book, lesson 1
- doc(learn.predict) - use in Jupyter to see function description and a source link
    - Book, lesson 1
- Other things in lesson 1
    - Image Segmentation: Car vision: Classify trees, road, pedestrian, etc.
    - NLP: Predict movie review sentiment
    - Tabular data: Predict values in one column given the values of other columns
        - Needs you to define which columns are:
            - Categorical/Discrete: occupation, department, etc.
            - Continuous: age, temperature, income, etc.
        - Can define data cleaning methods for fast.ai to do, like: Categorfiy,
        FillMissing, Normalize
        - Pre-trained models for tabular data don't exist (generally). Commonly use
        fit_one_cycle() to train from scratch.
        - Straight data example predicting income
    - Collaborative Filtering: Specific tabular data example. This one is for movies to
    recommend
        - Predicts rating users would give to movies
        - Example: Not a pre-trained model, but used fine_tune() over fit_one_cycle(),
        sometimes you can still use fine_tune() when not a pre-trained model. Takes a
        little trial and error to figure out.
- Concept: Do rapid prototyping and training on a subset of the data for fast results
until you have a good idea of what you have to do. Then scale up to use the full
dataset. Versus using the full dataset right off the bat. (time trade off, if it's fast
to use the full dataset, then by all means use full dataset)
    - Book, lesson 1
- Hyperparameters: Parameters about parameters. Examples: learning rates, data
augmentation strategies, network architecture, etc.
    - Book, lesson 1
- High Cardinality: lots of unique values (zip codes)
    - Book, lesson 2
- Not deep learning, but machine learning: random forests, gradient boosting
    - Book, lesson 2
- Deep learning can do recommendation systems but has an issue, you buy a book and it
suggests other forms of that books you already bought (nearly all machine learning
alorithms for recommendation systems have this issue)
    - Book, lesson 2
- NLP is good for dequences of discrete tokens with complex relationships and meaning
throughout the sequence. Like protein chains.
    - Book, lesson 2
- Python tip, map(): list_of_items_variable.map(function) calls the function passed to
map on each of the items in the list.
    - Book, lesson 2
- Confusion Matrix: Plot of the actual labels vs the predicted labels from a model.
One way of inspecting model performance visually. Only useful when you are predicting
categories.

Questions:
Lesson 2: When we get to training the bear image classifier we do this in the code:
```
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)
```
Shouldn't we put the RandomResizedCrop() in the batch transforms? Why do we have it in
the item transforms?