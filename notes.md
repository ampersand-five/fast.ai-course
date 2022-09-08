- Stochastic Gradient Descent, SGD: Simply the process of updating the parameters
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
    - Think of it like fitting a line to a quadratic set of data and when you're off you
    adjust the quadratic values to line it up better with the data and you keep doing
    that until it lines up with the data. The loss function is a function to tell you
    how far the line you're fitting, is, from the actual data. Larger it is, further
    away from the data the line is. Make it small to fit the data better. Example:
    https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work
        - Video, lesson 3, timestamp: 30:00
    - Different functions you can use for it
        - Mean Squared Error, mse: ((predictions - actuals)^2).mean()
            - Always positive, increasingly better as it approaches zero
- Derivative is the slope of a line at a given point. **ALSO KNOWN AS** the gradient
- Gradient: derivative, the slope of a line at a given point
- Gradient Descent: tying together the gradient/derivative concept and the loss concept,
it's a function that will change parameters to lower the loss which is just using the
slope the derivative found and following it down until it can get as close to zero and
bottom out, as it can, for each parameter, which in turn means that you've minimized the
loss and come as close as you can to fitting the function/neural net to the data/task.
    - Video, lesson 3, timestamp 33:00
- Integral is the total area under a line, either in whole or from point to point
- Rectified Linear (Unit) function, ReLU: linear function but make anything negative
just zero. Just a simple quadratic y=mx+b function.
    - Video, lesson 3, timestamp 33:00
    - Summing up infinite ReLU's allow you to map to anything in any number of
    dimensions. This is what neural nets do. The linear functions are y=mx+b format.
        - Video, lesson 3, timestamp 43:00
- Learning Rate: 
    - Used in gradient descent. After you get the gradient values of the
    random coefficients, (negative gradients mean you should increase the coefficient,
    positive gradients mean you should decreaese the coefficient), you can [subtract from the
    coefficients the [gradient values times the learning rate]]: coeffs-(gradients*LR)
        - Video, lesson 5, timestamp 40:00
    - How quickly or slowly you want the model to change it's coefficient values. Too
    small and it will take forever to learn, too large and you will overshoot (diverge)
    the value you're trying to reach and never get it.

- Feature engineering example: video lesson 3, timestamp: 1:04:29
- Metric: Measuring model performance. Function to measure quality of model's
predictions, printed after each epoch. Optimized for human consumption to have
visibility into the training of the model and having a feel for how well it's doing.
    - Error Rate: percentage of values being predicted/classified correctly
    - Accuracy: 1.0 - error_rate
    - Book, lesson 1
    - fast.ai/2019/09/24/metrics/
    - Examples:
        - Pearson Correlation Coefficient
            - Video lesson 4, timestamp: 1:04:12
- Loss vs Metric
    - Close, but different. Can be the same function used for both, or not.
    - Loss is for SGD to use to update weights, machine oriented
    - Metric is for humans to understand how model is performing
    - Book, lesson 1
- Dependent variable(s): Also known as the targets, labels, correct labels, y (vertical).
The variable you're trying to predict. Think: It's dependent on you predicting it.
    - Independent variable(s) is x (horizontal) (Book, lesson 2).
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
    - Also known as hidden layers, how many epochs is how many hidden layers
        - Each set of multiplying the coefficents by the values and summing it up
        - Video, lesson 5, timestamp: 1:04:40, 1:08:00
- The term "hidden layers" and "layer" both seem to be used for a couple purposes:

    Layer:
    - Each individual activation
    - Each individual epoch

    Hidden Layers:
    - The number of activations in each layer
    - The number of epochs

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
- fast.ai learn.predict() returns: prediction, index of predicted category, and array of
probabilities the thing is for each category
    - Class, lesson 2
- How many epochs to train for? Up to you, watch the error rate and make sure it's not
getting worse.
    - Class, lesson 2, timestamp: 53:20
- Production considerations:
    - Multiple versions of a model
    - A/B testing
    - Canarying
    - Refreshing data
    - Growing data
    - Handling data labeling
    - Monitoring everything
    - Detecting model rot
    - Domain shift: when the data our model sees over time changes and no longer matches
    what it was trained on
    - Lots more
    - Book, lesson 2
- When deploying to prod, make a slow roll-out paired with all human reviews of
predictions. Think about how the model could go wrong. A bear detection model trained
on a Google image search will not be able to recognize the bears seen in security
cameras around a camp. The images will be vastly different, framing, coloring,
bluriness, resolution, etc. Roll out model to just one camp, instead of all. Have the
model initially just highlight bears in red for a park ranger to then review. Have
reporting, like bear sightings doubled or halved at a park after putting the model in.
These would be signs that something might be wrong.
    - Book, lesson 2
- Convnext_<size> image models are as of 2022 great image models to use
    - Video, lesson 3, timestamp: 18:50
- Tensor: Array. But doesn't just work with numbers, also works with vectors of numbers
(other lits/arrays)
    - Can have multidimensional
        - 1-D tensor is just an array
        - 2-D tensor is rectangle of numbers or a table of numbers
        - 3-D tensor is layers of tables of numbers
        - etc.
        - Also called ranks instead of 1D,2D,3D -> Rank 1 tensor, Rank 2 tensor, etc.
        - Scalar is Rank 0 (just one number, x=tensor(2.0) is a scalar) || [1]
        - Vector is Rank 1 [1,1,1,1]
        - Matrix is Rank 2 [[1,1,1,1],
                            [1,1,1,1]]
        - Others are just referred to by rank
    - Video, lesson 3, timestamp: 36:00
- Neural net: using multiple column vectors of coefficients, aka activations, in one
epoch
    - Deep learning model: using multiple epochs where each epoch has many activations
- Activations: The arrays of coefficients we're multiplying into the data. Each column
vector of coefficients is one activation. More of them are activations.
    - Video, lesson 5, timestamp: 1:04:00
- Activation functions: the ReLU and Sigmoid functions we add at the end to clean up
the matrix multiplications we just finished doing.
    - Reminder: ReLU makes all negatives into 0, Sigmoid moves all predictions into a
    defined range, like 0 to 1, or -1 to 1, or 0 to 100.
    - Video, lesson 5, timestamp: 1:11:00
- Validation Sets: fast.ai/2017/11/13/validation-sets/
- When you have a binary, dependent variable, the final step that helps a lot is to put
it through a Sigmoid function.
    - Function: 1/(1+exp(-x)), one over one e to the negative x
    - Video, lesson 5, timestamp: 53:20
- The neural net and deep learning models didn't make much of a difference in the
Titanic dataset. This was because that data was small and tabular. Generally, you can
chuck a deep learning model at image and NLP problems and get a great result. But
tabular data isn't so easy. Pre-trained models don't exist for tabular data because they
are all unique and you have to think about the feature engineering for the data, a lot
more.
- Do something simple first on data like random forests to get a base line to compare
deep learning models to
- OneR: One Rule
    - Machine learning model.
    - A classifier that classifies based on just one attribute.
    - In the Titanic example it's like predicting survival using just the gender of the
    passengers. It was found to be one of the best machine learning classifiers for a
    wide range of real world datasets. Meaning, you should always make a OneR model as a
    baseline to compare to before getting more complicated things going.
    - A decision tree with a single binary split.
    - TwoR model is two rules, or two splits instead of one.
    - Video, lesson 5, timestamp: 1:42:48
- Decision Tree classifier:
    - A bunch of binary splits of the data to predict outcomes.
    - Only cares about ordering of the data
        - Does not need things normalized like the log of stuff and is not affected by
        outliers
    - Video, lesson 6, timestamp 13:00
    - Categorical variables tend to work fine as numbers in decision trees
- Random Forrest classifier:
    - Ensemble of decision trees
    - Uses the average of their predictions, called bagging
    - If the models are not correlated to each other, then the outcome for each decision
    tree will fall on a bell curve centered around the right answer. Some predictions
    will be too low and some predictions will be too high. But the average of all of
    the predictions will be zero, aka, the right answer.
        - Video, lesson 6, timestamp: 18:32







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